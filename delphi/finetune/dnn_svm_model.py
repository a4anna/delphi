
import copy
import io
import os
import glob
import time
import threading
import pickle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
import numpy as np
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as vision_models
from logzero import logger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.finetune.finetune_trainer_base import FinetuneTrainerBase
from delphi.finetune.finetune_model_base import FinetuneModelBase
from delphi.finetune.resnet import ResNet
from delphi.utils import get_weights, AverageMeter, Dict2Obj, ImageFromList
from dask_ml.wrappers import ParallelPostFit


TEST_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 32

RESNET_TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_lock = threading.Lock()
current_model = None
prev_state = None

class DnnSvmModel(FinetuneModelBase):

    def __init__(self, version: int, epoch: int,
                 optimizer_dict: Any, svc: Any = None, train_examples: Dict = {}):

        super().__init__(RESNET_TEST_TRANSFORMS, TEST_BATCH_SIZE, version)

        global model_lock, current_model, prev_state
        with model_lock:
            self._model = current_model
            self._model.eval()
            prev_state = self

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._version = version

        # These are just kept in case we want to resume training from this model. They're not actually necessary
        # for inference
        self.epoch = epoch
        self.optimizer_dict = optimizer_dict
        self.svc = svc
        self.train_examples = train_examples
        logger.info("NEW MODEL \n Version:{} \n SVC: {} \n TrainExamples{}".format(self.version,
            self.svc is not None, self.train_examples))
        m_params = self._model.state_dict()
        keys = list(m_params.keys())[-2]
        logger.info("Predict model param {} {}".format(keys, m_params[keys][0, :5]))

    def get_predictions(self, inputs: torch.Tensor) -> List[float]:
        with torch.no_grad():
            # inputs = inputs.to(self.device)
            with model_lock:
                self._model = current_model
                self._model.to(self.device)
                self._model.eval()
                svc = self.svc
                output = self._model(inputs)
            if svc is None:
                probability = torch.softmax(output[0], dim=1)
                probability = np.squeeze(probability.cpu().numpy())[:, 1]
                return probability
            else:
                features = output[1]
                features = features.data.cpu().numpy()
                try:
                    result = svc.predict_proba(features)[:, 1]  # self.svc.decision_function(features)
                except Exception as e:
                    logger.error(e)

                return result

    def get_bytes(self) -> bytes:
        bytes = io.BytesIO()
        torch.save({
            'epoch': self.epoch,
            'state_dict': self._model.state_dict(),
            'optimizer': self.optimizer_dict,
            'clf': pickle.dumps(self.svc.estimator) if (self.svc is not None) else None,
            'train_examples': self.train_examples
        }, bytes)
        bytes.seek(0)

        return bytes.getvalue()


class DnnSvmTrainer(FinetuneTrainerBase):
    """
    Trains Resnet and SVM models:
    Finetunes resnet after an increment of 50 positives
    Trains SVM after 10% increment of positives
    """

    def __init__(self, context: ModelTrainerContext, trainer_start: int):
        super().__init__(context, trainer_start)

        self._curr_epoch = 0
        self._global_step = 0
        self.train_positives = {'finetune': 0, 'svm': 0}
        self.curr_positives = 0
        self.finetune_first = True

        self.resnet_train = 0.5
        self.svm_train = 0.1
        self.conditions_train = {'finetune': (self.relative_condition, ('finetune', self.resnet_train)),
                           'svm': (self.relative_condition, ('svm', self.svm_train))}

        params = {
            'model_arch': 'resnet50',
            'num_cats': 2,
            'feature_extractor': False,
            'lr': 0.002,
            'decay_epochs': 50,
            'weight_decay': 1e-04,
            'momentum': 0.9,
            'pretrained': True,
        }

        self.params = Dict2Obj(params)
        self.extractor = self.params.feature_extractor

        self._model = ResNet(self.params).to(self.device)
        # self._model = torch.nn.DataParallel(self._model).to(self.device)

        # self._model.train()

        logger.info("RESNET TRAINER CALLED")

        # setup optimizer
        self._optimizer = torch.optim.SGD(
                            filter(lambda p: p.requires_grad, self._model.parameters()),
                            lr=self.params.lr,
                            momentum=self.params.momentum,
                            weight_decay=self.params.weight_decay)

        self._criterion = nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        self._train_transforms = transforms.Compose([
            transforms.Resize(259),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.previous = None

    def increment_condition(self, key, num):
        if (self.curr_positives - self.train_positives[key]) >= num:
            return True
        return False

    def relative_condition(self, key, percent):
        if (self.curr_positives - self.train_positives[key]) >= (percent * self.train_positives[key]):
            return True
        return False

    def adjust_learning_rate(self, epoch):
        """Decays the learning rate"""
        lr = self.params.lr * (0.1 ** (epoch // self.params.decay_epochs))
        print('learning rate', lr)
        for param_group in self._optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr


    def load_from_file(self, model_version: int, file: bytes) -> Model:
        bytes = io.BytesIO()
        bytes.write(file)
        bytes.seek(0)
        checkpoint = torch.load(bytes)
        self._model = ResNet(self.params).to(self.device)
        self._model.load_state_dict(checkpoint['state_dict'])
        self._curr_epoch = checkpoint['epoch']
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        clf = checkpoint['clf']
        if clf is not None:
            clf = ParallelPostFit(pickle.loads(checkpoint['clf']))
        train_examples = checkpoint['train_examples']

        global model_lock, current_model
        with model_lock:
            prev_model = None
            if current_model is not None:
                prev_model = current_model.to('cpu')
            del prev_model
            model_pred = copy.deepcopy(self._model)
            model_pred.eval()
            model_pred.to(self.device)
            current_model = model_pred

        return ResNetModel(self.get_new_version(), self._curr_epoch, self._optimizer.state_dict(), clf, train_examples)

    def train_svm(self, len_positives, train_dir: Path) -> Model:
        logger.info("Training SVM")
        global model_lock, current_model
        # with model_lock:
        #     self._model = current_model
        # self._model.to('cpu')
        logger.info("Model none ? {}".format(self._model is None))

        self._model.eval()

        image_list = []

        for label in ['0', '1']:
            image_list.extend(glob.glob(os.path.join(str(train_dir), label, "*")))

        # dataset = datasets.ImageFolder(str(train_dir), transform=self._train_transforms)
        dataset = ImageFromList(image_list, transform=RESNET_TEST_TRANSFORMS, limit=(10*len_positives))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=4)

        # max_counts = {'0': len_positives*10, '1': len_positives}
        num_positives = sum(dataset.targets)
        total_train_number = len(dataset.targets)
        curr_counts = {'0': (total_train_number - num_positives), '1': num_positives}
        X = []
        y = []

        with model_lock:
            for i, (inputs, targets) in enumerate(train_loader):
                # measure data loading time
                inputs = inputs.to(self.device, non_blocking=True)
                # targets = targets.to(self.device, non_blocking=True)

                # compute output
                outputs = self._model(inputs)
                features = outputs[1]
                features = features.data.cpu().numpy()
                targets = targets.data.numpy()

                for label, feature in zip(targets, features):
                    # label = int(label)
                    # if curr_counts[str(label)] > max_counts[str(label)]:
                    #     continue
                    # curr_counts[str(label)] += 1
                    y.append(label)
                    X.append(feature)


        X = np.array(X)
        y = np.array(y)

        param = {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
        estimator = SVC(random_state=42, probability=True, class_weight='balanced', **param)
        # clf = estimator
        weights = get_weights(dataset.targets)
        sampling = min(1.0, weights[0]/weights[1])
        # clf = ParallelPostFit(BalancedBaggingClassifier(base_estimator=estimator,
        #                                                     max_samples=0.8,
        #                                                     sampling_strategy='auto',
        #                                                     n_jobs=4, random_state=42))
        clf = ParallelPostFit(estimator)
        clf.fit(X, y)
        torch.cuda.empty_cache()

        with model_lock:
            if current_model is None:
                model_pred = copy.deepcopy(self._model)
                model_pred.eval()
                current_model = model_pred


        return DnnSvmModel(self.get_new_version(), self._curr_epoch, self._optimizer.state_dict(), clf, curr_counts)

    def train_model(self, train_dir: Path) -> Model:
        global model_lock, current_model, prev_state
        len_positives = len(glob.glob(os.path.join(str(train_dir), '1', '*')))
        labels = []

        while '0' not in labels:
            labels = [os.path.basename(path) for path in glob.glob(os.path.join(str(train_dir), '*'))]
            time.sleep(30)


        # epochs = 5
        # params = {
        #     'model_arch': 'resnet50',
        #     'num_cats': 2,
        #     'feature_extractor': False,
        #     'lr': 0.001,
        #     'decay_epochs': 5,
        #     'weight_decay': 1e-04,
        #     'momentum': 0.9,
        #     'pretrained': True,
        # }

        # self.params = Dict2Obj(params)

        self.curr_positives = len_positives

        # CHECK if any train condition is satisfied
        logger.info("Number of positives {}".format(len_positives))

        def check_train_condition():
            for key in self.conditions_train:
                func_, args = self.conditions_train[key]
                if func_(*args):
                    return key
            return None

        condition = check_train_condition()

        # if condition is None:
        #     logger.info("[FINETUNE] No training")
        #     with model_lock:
        #         return ResNetModel(prev_state.version, prev_state.epoch, prev_state.optimizer_dict,
        #                            prev_state.svc, prev_state.train_examples)

        if condition == "finetune" and self.previous == "finetune":
            condition = "svm"

        if self.train_positives['finetune'] == 0:
            logger.info("[FINETUNE] FIRST TRAINING")
            condition = "finetune"
            self.train_positives['finetune'] = len_positives
        logger.info("[FINETUNE] CONDITION {}".format(condition))

        if condition != "finetune":
            logger.info("[FINETUNE] SVM training")
            self.train_positives['svm'] = len_positives
            self.previous = "svm"
            return self.train_svm(len_positives, train_dir)

        self.previous = "finetune"
        torch.cuda.empty_cache()
        logger.info("[FINETUNE] ResNet training")

        self._model = ResNet(self.params)
        self._model.to(self.device)

        if len_positives < 50:
            epochs = 2
        elif len_positives < 100:
            epochs = 5
        else:
            epochs = 10

        # setup optimizer
        self._optimizer = torch.optim.SGD(
                            filter(lambda p: p.requires_grad, self._model.parameters()),
                            lr=self.params.lr,
                            momentum=self.params.momentum)

        self._criterion = nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        # scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=7, gamma=0.1)

        self._model.train()

        start_time = time.time()
        start_epoch = self._curr_epoch
        end_epoch = start_epoch + epochs
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # dataset = datasets.ImageFolder(str(train_dir), transform=self._train_transforms)
        image_list = []

        for label in train_dir.iterdir():
            paths = glob.glob(os.path.join(str(train_dir), label, "*"))
            logger.info("LABEL : {} LENGTH: {}".format(label, len(paths)))
            image_list.extend(paths)

        # dataset = datasets.ImageFolder(str(train_dir), transform=self._train_transforms)
        dataset = ImageFromList(image_list, transform=self._train_transforms, limit=500*len_positives)
        weights = get_weights(dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, sampler=sampler, num_workers=8)

        train_image_count = len(dataset.targets)
        curr_count = {'0': int(train_image_count - len_positives),
                      '1': len_positives}


        for epoch in tqdm(range(start_epoch, end_epoch)):
            self._model.train()
            self._curr_epoch += 1

            end = time.time()

            self.adjust_learning_rate(epoch)
            running_loss = 0.0
            running_acc = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                data_time.update(time.time() - end)
                # measure data loading time
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self._optimizer.zero_grad()
                # compute output
                outputs = self._model(inputs)
                outputs = outputs[0]
                _, preds = torch.max(outputs, 1)
                loss = self._criterion(outputs, targets)


                # compute gradient and do SGD step
                # self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                # measure accuracy and record loss
                losses.update(loss.item(), inputs.size(0))
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == targets.data)

                end = time.time()

                if i % 50 == 0:
                    # self.context.tb_writer.add_scalar(str('train/loss'), loss.item(),
                    #                                   self._curr_epoch * train_image_count + i)

                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f}))'
                          .format(epoch, i, train_image_count, batch_time=batch_time, data_time=data_time, loss=losses))

            # scheduler.step()
            epoch_loss = running_loss / train_image_count
            epoch_acc = running_acc.double() / train_image_count
            print("LOSS {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
        end_time = time.time()
        logger.info('Trained model for {} epochs in {:.3f} seconds'.format(epochs, end_time - start_time))

        with model_lock:
            prev_model = None
            if current_model is not None:
                prev_model = current_model.to('cpu')
            model_pred = copy.deepcopy(self._model)
            model_pred.eval()
            current_model = model_pred
            m_params = self._model.state_dict()
            keys = list(m_params.keys())[-2]
            logger.info("Trained model param {} {}".format(keys, m_params[keys][0, :5]))
            del prev_model
            # del self._model
            torch.cuda.empty_cache()

        # if self.finetune_first: # np.all([v == 0 for _, v in self.train_positives.items()]):
        #     self.finetune_first = False
        #     logger.info("UNLINKING NEGATIVES")
        #     if os.path.exists(str(train_dir / '0')):
        #         for path in glob.glob(os.path.join(str(train_dir), '0', '*'))[6:]:
        #             path = Path(path)
        #             if path.exists():
        #                 path.unlink()
        #                 # path.rename(str(path).replace('/1/', '/2/'))

        # TODO REMOVE
        # if len_positives < 50:
        #     self.resnet_train = 50
        #     self.conditions_train['finetune'] = (self.increment_condition, ('finetune', self.resnet_train))
        # else:
        self.train_positives['finetune'] = len_positives
        # epochs = self._curr_epoch
        self._curr_epoch = 0

        return DnnSvmModel(self.get_new_version(), self._curr_epoch, self._optimizer.state_dict(), None, curr_count)


