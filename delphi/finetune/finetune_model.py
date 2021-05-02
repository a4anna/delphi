
import copy
import io
import os
import glob
import time
import threading
import pickle
from sklearn.svm import SVC
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

class ResNetModel(FinetuneModelBase):

    def __init__(self, version: int, epoch: int,
                 optimizer_dict: Any, svc: Any = None, train_examples: Dict = {}):

        super().__init__(RESNET_TEST_TRANSFORMS, TEST_BATCH_SIZE, version)

        global model_lock, current_model, prev_state
        with model_lock:
            self._model = current_model
            self._model.eval()
            prev_state = self


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
        with model_lock and torch.no_grad():
            # inputs = inputs.to(self.device)
            output = self._model(inputs)
            if self.svc is None:
                probability = torch.softmax(output[0], dim=1)
                probability = np.squeeze(probability.cpu().numpy())[:, 1]
                return probability
            else:
                features = output[1]
                features = features.data.cpu().numpy()
                return self.svc.decision_function(features)

    def get_bytes(self) -> bytes:
        bytes = io.BytesIO()
        torch.save({
            'epoch': self.epoch,
            'state_dict': self._model.state_dict(),
            'optimizer': self.optimizer_dict,
            'clf': pickle.dumps(self.svc)
        }, bytes)
        bytes.seek(0)

        return bytes.getvalue()


class ResNetTrainer(FinetuneTrainerBase):
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

        resnet_train = 50
        svm_train = 0.1
        self.conditions_train = {'finetune': (self.increment_condition, ('finetune', resnet_train)),
                           'svm': (self.relative_condition, ('svm', svm_train))}

        params = {
            'model_arch': 'resnet50',
            'num_cats': 2,
            'feature_extractor': False,
            'lr': 0.001,
            'decay_epochs': 20,
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

        global model_lock, current_model
        with model_lock:
            model_pred = copy.deepcopy(self._model)
            model_pred.eval()
            current_model = model_pred

        return ResNetModel(self.get_new_version(), self._curr_epoch, self._optimizer.state_dict())

    def train_svm(self, len_positives, train_dir: Path) -> Model:
        logger.info("Training SVM")
        global model_lock, current_model
        with model_lock:
            self._model = current_model
            self._model.eval()

        image_list = []

        for label in ['0', '1']:
            image_list.extend(glob.glob(os.path.join(str(train_dir), label, "*")))

        # dataset = datasets.ImageFolder(str(train_dir), transform=self._train_transforms)
        dataset = ImageFromList(image_list, transform=self._train_transforms)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=8)

        max_counts = {'0': len_positives*10, '1': len_positives}
        curr_counts = {'0': 0, '1': 0}
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
                    label = int(label)
                    curr_counts[str(label)] += 1
                    if curr_counts[str(label)] > max_counts[str(label)]:
                        continue
                    y.append(label)
                    X.append(feature)

                del inputs

        X = np.array(X)
        y = np.array(y)

        clf = SVC(random_state=42, probability=True, class_weight='balanced', kernel='linear')
        clf.fit(X, y)

        return ResNetModel(self.get_new_version(), 0, {}, clf, curr_counts)

    def train_model(self, train_dir: Path) -> Model:
        global model_lock, current_model, prev_state
        len_positives = len(glob.glob(os.path.join(str(train_dir), '1', '*')))

        epochs = 5
        params = {
            'model_arch': 'resnet50',
            'num_cats': 2,
            'feature_extractor': False,
            'lr': 0.001,
            'decay_epochs': 5,
            'weight_decay': 1e-04,
            'momentum': 0.9,
            'pretrained': True,
        }

        self.params = Dict2Obj(params)

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

        if condition is None:
            logger.info("[FINETUNE] No training")
            with model_lock:
                return ResNetModel(prev_state.version, prev_state.epoch, prev_state.optimizer_dict,
                                   prev_state.svc, prev_state.train_examples)

        if condition == "svm":
            logger.info("[FINETUNE] SVM training")
            return self.train_svm(len_positives, train_dir)

        torch.cuda.empty_cache()
        logger.info("[FINETUNE] ResNet training")

        self._model = ResNet(self.params).to(self.device)

        # setup optimizer
        self._optimizer = torch.optim.SGD(
                            filter(lambda p: p.requires_grad, self._model.parameters()),
                            lr=self.params.lr,
                            momentum=self.params.momentum)

        # self._optimizer = torch.optim.SGD(
        #                     filter(lambda p: p.requires_grad, self._model.parameters()),
        #                     lr=self.params.lr,
        #                     momentum=self.params.momentum,
        #                     weight_decay=self.params.weight_decay)

        self._criterion = nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=7, gamma=0.1)

        self._model.train()

        start_time = time.time()
        start_epoch = 0
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
        dataset = ImageFromList(image_list, transform=self._train_transforms, limit=5000)
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
            # self.adjust_learning_rate(epoch)

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

            scheduler.step()
            epoch_loss = running_loss / train_image_count
            epoch_acc = running_acc.double() / train_image_count
            print("LOSS {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
        end_time = time.time()
        logger.info('Trained model for {} epochs in {:.3f} seconds'.format(epochs, end_time - start_time))

        with model_lock:
            model_pred = copy.deepcopy(self._model)
            model_pred.eval()
            current_model = model_pred
            m_params = self._model.state_dict()
            keys = list(m_params.keys())[-2]
            logger.info("Trained model param {} {}".format(keys, m_params[keys][0, :5]))
            del self._model
            torch.cuda.empty_cache()

        # epochs = self._curr_epoch
        self._curr_epoch = 0

        return ResNetModel(self.get_new_version(), self._curr_epoch, self._optimizer.state_dict(), None, curr_count)


