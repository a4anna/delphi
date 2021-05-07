from collections import defaultdict
from logzero import logger
from google.protobuf.json_format import MessageToDict
from delphi.proto.learning_module_pb2_grpc import LearningModuleServiceStub


def get_stats(condition, *args):
    if condition == "time":
        return TimerStats(*args)
    elif condition == "model":
        return VersionStats(*args)
    else:
        raise NotImplementedError()


class DelphiStats(object):

    def __init__(self, stubs, search_id):
        self.stubs = stubs
        self.search_id = search_id

    def accumulate_search_stats(self):
        stats = defaultdict(lambda: 0)
        for stub in self.stubs:
            try:
                search_stat = stub.GetSearchStats(self.search_id)
                search_stat = MessageToDict(search_stat)
                # logger.info(search_stat)
                for k, v in search_stat.items():
                    if isinstance(v, dict):
                        others = v
                        for key, value in others.items():
                            stats[key] += float(value)
                    else:
                        stats[k] += float(v)
            except Exception as e:
                logger.error(e)
                pass

        logger.info(stats)
        return stats

    def get_latest_model_version(self):
        model_version = self.stubs[0].GetModelStats(self.search_id).version
        # model_version = max([stub.GetModelStats(self.search_id).version for stub in self.stubs])
        # logger.info("Model Version {}".format(model_version))

        return model_version


class TimerStats(DelphiStats):
    """
    Returns Search statistics at regular intervals
    """
    def __init__(self, stubs, search_id, stop_event, stats_queue, interval):
        super().__init__(stubs, search_id)
        self.stop_event = stop_event
        self.stats_queue = stats_queue
        self.interval = interval

    def start(self):
        # self.stats_queue.put(self.accumulate_search_stats())

        while not self.stop_event.wait(self.interval):
            stats = self.accumulate_search_stats()
            if stats:
                self.stats_queue.put(stats)

        self.stats_queue.put(None)


class VersionStats(DelphiStats):
    """
    Returns Search statistics at regular intervals
    """
    def __init__(self, stubs, search_id, stop_event, stats_queue, interval=1):
        super().__init__(stubs, search_id)
        self.stop_event = stop_event
        self.stats_queue = stats_queue
        self.interval = interval
        self.version = -1

    def start(self):
        ##self.stats_queue.put(self.accumulate_search_stats())

        while not self.stop_event.wait(timeout=self.interval):
            current_version = self.get_latest_model_version()
            if self.version != current_version:
                logger.info("Version not equal")
                self.version = current_version
                stats = self.accumulate_search_stats()
                if stats:
                    stats['version'] = self.version
                    self.stats_queue.put(stats)

        self.stats_queue.put(None)
