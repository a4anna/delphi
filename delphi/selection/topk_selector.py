import copy
import os
import math
import queue
import time
import threading
from collections import defaultdict
from logzero import logger
from typing import Optional
from pathlib import Path

from delphi.model import Model
from delphi.result_provider import ResultProvider
from delphi.selection.reexamination_strategy import ReexaminationStrategy
from delphi.selection.selector_base import SelectorBase
from delphi.selection.selector_stats import SelectorStats
from delphi.utils import get_example_key
from opendiamond.server.object_ import ATTR_DATA

class TopKSelector(SelectorBase):

    def __init__(self, k: int, batch_size: int,
                 reexamination_strategy: ReexaminationStrategy,
                 add_negatives: bool = True):
        assert k < batch_size
        super().__init__()

        self._k = k
        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0
        self._insert_lock = threading.Lock()
        self.last_result_time = None
        self.timeout = 300
        self.add_negatives = add_negatives
        self.easy_negatives = defaultdict(list)
        self.version = -1

    def result_timeout(self, interval=0):
        if interval == 0 or self.last_result_time is None:
            return False
        return (time.time() - self.last_result_time) >= interval

    def add_result_inner(self, result: ResultProvider) -> None:
        with self._insert_lock:
            if '/1/' in result.id:
                logger.info("Queueing {} Score {}".format(result.id, result.score))
            self._priority_queues[-1].put((-result.score, result.id, result))
            self._batch_added += 1
            # if self._batch_added == self._batch_size or \
            #     (self.result_timeout(self.timeout) and self._batch_added > 0):
            if self._batch_added == self._batch_size:
                for _ in range(self._k):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._batch_added = 0
                self.last_result_time = time.time()

    def add_easy_negatives(self, path: Path) -> None:
        if not self.add_negatives:
            return

        negative_path = path / '2'
        os.makedirs(str(negative_path), exist_ok=True)

        result_list = []
        result_queue = queue.PriorityQueue()
        with self._insert_lock:
            result_queue.queue = copy.deepcopy(self._priority_queues[-1].queue)

        result_list = [item[-1] for item in list(result_queue.queue)]
        length_results = len(result_list)

        if length_results < 10:
            return
        result_list = sorted(result_list, key= lambda x: x.score, reverse=True)

        num_auto_negative = int(0.25 * length_results)
        logger.info("[EASY NEG] Length of result list {} {}".format(length_results, num_auto_negative))
        auto_negative_list = result_list[-num_auto_negative:]

        for result in auto_negative_list:
            object_id = result.id
            example = self._search.retriever.get_object(object_id, [ATTR_DATA])
            example_file = get_example_key(example.content)
            example_path = negative_path / example_file
            with example_path.open('wb') as f:
                f.write(example.content)
            with self._insert_lock:
                self.easy_negatives[self.version].append(example_path)

    def new_model_inner(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            if model is not None:
                version = self.version
                if version in self.easy_negatives:
                    versions = [v for v in self.easy_negatives.keys() if v <= version]
                    for v in versions:
                        self.delete_examples(self.easy_negatives[v])

                self.version += 1
                # add fractional batch before possibly discarding results in old queue
                for _ in range(math.ceil(float(self._k) * self._batch_added / self._batch_size)):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._priority_queues = self._reexamination_strategy.get_new_queues(model, self._priority_queues)
            else:
                # this is a reset, discard everything
                self._priority_queues = [queue.PriorityQueue()]

            self._batch_added = 0

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            items_processed = self.items_processed

        return SelectorStats(items_processed, 0, None, 0)
