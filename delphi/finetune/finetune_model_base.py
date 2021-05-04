import io
import multiprocessing as mp
from abc import abstractmethod
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List, Callable, Iterable, Tuple
from logzero import logger

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from delphi.model import Model
from delphi.object_provider import ObjectProvider
from delphi.result_provider import ResultProvider
from delphi.utils import log_exceptions, bounded_iter
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess(request: ObjectProvider) -> Tuple[ObjectProvider, torch.Tensor]:
    try:
        image = Image.open(io.BytesIO(request.content))

        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        image = Image.fromarray(np.random.randint(0,255,(256,256,3)),'RGB')

    return request, get_test_transforms()(image)


class FinetuneModelBase(Model):
    test_transforms: transforms.Compose

    def __init__(self, test_transforms: transforms.Compose, batch_size: int, version: int):
        self._batch_size = batch_size
        self._test_transforms = test_transforms
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._version = version

    @property
    def version(self) -> int:
        return self._version

    @abstractmethod
    def get_predictions(self, input: torch.Tensor) -> List[float]:
        pass

    # @abstractmethod
    # def get_predictions(self, inputs: torch.Tensor) -> List[float]:
    #     logger.error("NOT HERE")
    #     pass
        # assert self.model is not None
        # with torch.no_grad():
        #     outputs = self.model(inputs)
        #     probability = torch.nn.functional.softmax(outputs, dim=1)
        #     probability = np.squeeze(probability.cpu().numpy())[:, 1]
        #     return probability

    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        semaphore = mp.Semaphore(256)  # Make sure that the load function doesn't overload the consumer
        batch = []

        with mp.get_context('spawn').Pool(min(16, mp.cpu_count()), initializer=set_test_transforms,
                                          initargs=(self._test_transforms,)) as pool:
            items = pool.imap_unordered(preprocess, bounded_iter(requests, semaphore))

            for item in items:
                semaphore.release()
                batch.append(item)
                if len(batch) == self._batch_size:
                    yield from self._process_batch(batch)
                    batch = []

        if len(batch) > 0:
            yield from self._process_batch(batch)

    def infer_dir(self, directory: Path, callback_fn: Callable[[int, float], None]) -> None:
        dataset = datasets.ImageFolder(str(directory), transform=self._test_transforms)
        data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=mp.cpu_count())

        with torch.no_grad():
            for inputs, target in tqdm(data_loader):
                predictions = self.get_predictions(inputs.to(self._device, non_blocking=True))
                del inputs

                for i in range(len(predictions)):
                    callback_fn(target[i].tolist(), predictions[i])

    @property
    def scores_are_probabilities(self) -> bool:
        return True

    def _process_batch(self, batch: List[Tuple[ObjectProvider, torch.Tensor]]) -> Iterable[ResultProvider]:
        tensors = torch.stack([f[1] for f in batch]).to(self._device, non_blocking=True)
        predictions = self.get_predictions(tensors)
        del tensors
        for i in range(len(batch)):
            score = predictions[i]
            yield ResultProvider(batch[i][0].id, '1' if score >= 0.5 else '0',
                                 score, self.version,
                                 batch[i][0].attributes, batch[i][0].gt)
            # if self.svc is None:
            #     yield ResultProvider(batch[i][0].id, '1' if score >= 0.5 else '0',
            #                          score, self.version,
            #                          batch[i][0].attributes, batch[i][0].gt)
            # else:
            #     yield ResultProvider(batch[i][0].id, '1' if score >= 0 else '0',
            #                          score, self.version,
            #                          batch[i][0].attributes, batch[i][0].gt)



_test_transforms: transforms.Compose
_semaphore: mp.Semaphore


def get_test_transforms() -> transforms.Compose:
    global _test_transforms
    return _test_transforms


@log_exceptions
def set_test_transforms(test_transforms: transforms.Compose) -> None:
    global _test_transforms
    _test_transforms = test_transforms
