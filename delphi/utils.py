import hashlib
import multiprocessing as mp
import random
import threading
from functools import wraps
from queue import Queue
from typing import List, Union, Iterable, Any, TypeVar
import torch
import numpy as np
from logzero import logger
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

class ImageFromList(torch.utils.data.Dataset):
    """Load dataset from path list"""
    def __init__(self, image_list, transform, label_list=None, limit=None):
        self.loader = self.image_loader
        if label_list is None:
            random.shuffle(image_list)
        # self.imlist = image_list
        self.transform = transform
        target_transform = lambda x: 1 if '/1/' in x else 0
        labels = [target_transform(path) for path in image_list]
        # self.imlist = [p.split('@')[0] for p in image_list]
        self.classes = sorted(set(labels))
        if not label_list:
            label_list = labels

        if limit is None:
            limit = len(labels)

        max_count = {k: limit for k in set(label_list)}
        num_count = {k: 0 for k in max_count}
        self.targets = []
        self.imlist = []
        for target, img in zip(label_list, image_list):
            num_count[target] += 1
            if num_count[target] > max_count[target]:
                continue
            self.targets.append(target)
            self.imlist.append(img)

        #     label_list = [self.classes.index(label) for label in labels]
        self.targets = label_list

    def image_loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        impath = self.imlist[idx]
        target = self.targets[idx]
        img = self.loader(impath)
        img = self.transform(img)
        #basepath = impath.split('/')[-1]
        return img, target

    def __len__(self):
        return len(self.imlist)


def get_example_key(content) -> str:
    return hashlib.sha1(content).hexdigest() + '.jpg'


def get_weights(targets: List[int], num_classes=2) -> List[int]:
    class_weights = [0] * num_classes
    classes, counts = np.unique(targets, return_counts=True)
    for i in range(len(classes)):
        class_weights[classes[i]] = len(targets) / float(counts[i])

    logger.info('Class weights: {}'.format(class_weights))

    weight = [0] * len(targets)
    for idx, val in enumerate(targets):
        weight[idx] = class_weights[val]

    return weight


T = TypeVar('T')


def bounded_iter(iterable: Iterable[T], semaphore: threading.Semaphore) -> Iterable[T]:
    for item in iterable:
        semaphore.acquire()
        yield item


def to_iter(queue: Union[Queue, mp.Queue]) -> Iterable[Any]:
    def iterate():
        while True:
            example = queue.get()
            if example is None:
                break
            yield example

    return iterate()


def log_exceptions(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e

    return func_wrapper

