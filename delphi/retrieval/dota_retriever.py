import io
import queue
import os
import random
import time
import threading
from pathlib import Path
from typing import Iterable, Sized
from collections import defaultdict

from logzero import logger
from opendiamond.attributes import StringAttributeCodec
from opendiamond.client.search import DiamondSearch, FilterSpec, Blob, _DiamondBlastSet
from opendiamond.config import DiamondConfig
from opendiamond.protocol import XDR_reexecute
from opendiamond.scope import ScopeCookie
from opendiamond.server.object_ import ATTR_OBJ_ID, ATTR_DATA, ATTR_DEVICE_NAME

from delphi.object_provider import ObjectProvider
from delphi.proto.learning_module_pb2 import DelphiObject, DiamondDataset
from delphi.retrieval.diamond_attribute_provider import DiamondAttributeProvider
from delphi.retrieval.retriever import Retriever
from delphi.retrieval.retriever_stats import RetrieverStats

ATTR_GT_LABEL = '_gt_label'  # attr of ground truth label
STRING_CODEC = StringAttributeCodec()


class DotaRetriever(Retriever):

    def __init__(self, dataset: DiamondDataset):
        self._dataset = dataset
        # self._search = self._create_search()
        self._start_event = threading.Event()
        self._command_lock = threading.RLock()
        stats_keys = ['total_objects', 'total_images', 'dropped_objects',
                      'false_negatives', 'retrieved_images', 'retrieved_tiles']
        self._stats = {x: 0 for x in stats_keys}
        self._running = False
        self.timeout = 20
        self._start_time = time.time()
        self.result_queue = queue.Queue()
        files = sorted(open("/srv/diamond/INDEXES/GIDIDXDELPHI").read().splitlines())
        self.img_tile_map = defaultdict(list)
        for f in files:
            key = os.path.basename(f).split('_')[0]
            self.img_tile_map[key].append(f)
        self.images = sorted(list(self.img_tile_map.keys()))
        random.shuffle(self.images)
        self._stats['total_objects'] = len(files)
        self._stats['total_images'] = len(self.images)
        threading.Thread(target=self.stream_objects, name='stream').start()

        try:
            self._diamond_config = DiamondConfig()
        except Exception as e:
            logger.info('No local diamond config found')
            logger.exception(e)
            self._diamond_config = None

    def stream_objects(self):

        if not self._running:
            return
        self._start_time = time.time()
        for key in self.images:
            with self._command_lock:
                self._stats['retrieved_images'] += 1
                logger.info("Retrieved {} @ {}".format(key, time.time() - self._start_time))
            tiles = self.img_tile_map[key]
            # logger.info(key)

            for tile in tiles:
                image_path = Path(os.path.join(self._diamond_config.dataroot, tile))
                with open(image_path, 'rb') as f:
                    content = f.read()

                object_id = "http://localhost:5873/collection/id/"+tile
                result = {
                    'Device-Name': STRING_CODEC.encode(self._diamond_config.serverids[0]),
                    '_ObjectID': STRING_CODEC.encode(object_id),
                }
                with self._command_lock:
                    self._stats['retrieved_tiles'] += 1

                self.result_queue.put(ObjectProvider(object_id, content, DiamondAttributeProvider(result, image_path), False))
            time.sleep(self.timeout)

        # self.result_queue.put(None)
        self._running = False

    def is_running(self):
        return self._running

    def start(self) -> None:
        with self._command_lock:
            self._running = True
            self._start_time = time.time()

        self._start_event.set()
        threading.Thread(target=self.stream_objects, name='stream').start()


    def stop(self) -> None:
        with self._command_lock:
            self._final_stats = self.get_stats()
            self._running = False

    def get_objects(self) -> Iterable[ObjectProvider]:
        while True:
            try:
                result = self.result_queue.get()
                yield result
            except Exception:
                time.sleep(self.timeout)
                continue

    def get_object(self, object_id: str, attributes: Sized) -> DelphiObject:
        if not self._running:
            return None
        object_path = object_id.split("collection/id/")[-1]
        image_path = os.path.join(self._diamond_config.dataroot, object_path)
        with open(image_path, 'rb') as f:
            content = f.read()

        # Return object attributes
        dct = {
                'Device-Name': STRING_CODEC.encode(self._diamond_config.serverids[0]),
                '_ObjectID': STRING_CODEC.encode(object_id),
              }

        return DelphiObject(objectId=object_id, content=content, attributes=dct)

    def get_stats(self) -> RetrieverStats:
        self._start_event.wait()

        with self._command_lock:
            stats = self._stats.copy()

        return RetrieverStats(stats)
