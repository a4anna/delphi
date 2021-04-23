import grpc
import json
import os
import shutil
import signal
import sys
import yaml
import uuid
import time
import threading
import multiprocessing_logging
from pathlib import Path
from subprocess import call
from logzero import logger
import queue
from delphi.utils import to_iter
import random
import numpy as np
from datetime import datetime
from stats import get_stats
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToJson
from itertools import cycle

from delphi.proto.learning_module_pb2 import InferRequest, InferResult, ModelStats, \
    ImportModelRequest, ModelArchive, LabeledExampleRequest, Filter, SearchId, StringRequest,\
    AddLabeledExampleIdsRequest, LabeledExample, DelphiObject, GetObjectsRequest, SearchStats, SearchInfo, \
    CreateSearchRequest, Dataset, DiamondDataset, ReexaminationStrategyConfig, NoReexaminationStrategyConfig

from delphi.proto.learning_module_pb2 import ModelConditionConfig, ExamplesPerLabelConditionConfig, \
     ModelConfig, SVMConfig, SVMMode, PercentageThresholdPolicyConfig, ExampleSetWrapper,\
     SelectorConfig, RetrainPolicyConfig, TopKSelectorConfig, ThresholdConfig, ExampleSet

from delphi.proto.learning_module_pb2_grpc import LearningModuleServiceStub

from opendiamond.server.filter import ATTR_FILTER_SCORE
from opendiamond.attributes import StringAttributeCodec

STRING_CODEC = StringAttributeCodec()
INDEX_PATH = "/srv/diamond/INDEXES/GIDIDXDELPHI"


class DelphiClient(object):

    def __init__(self, config):
        multiprocessing_logging.install_mp_handler()
        self.config = config
        port = self.config['port']

        def get_channel(host, port):
            print("{}:{}".format(host, port))
            return grpc.insecure_channel('{}:{}'.format(host, port))

        self.hosts = self.config['nodes']
        print(self.hosts)
        self.nodes = ["{}:{}".format(host, port) for host in self.hosts]
        self.stubs = [LearningModuleServiceStub(get_channel(host, port)) for host in self.hosts]
        # self.stubs = [LearningModuleServiceStub(node) for node in self.nodes]

        for stub in self.stubs:
            stub.GetMessage(StringRequest(msg="Client Says Hi"))
        # random seed, train dir and output dir setup
        experiment_params = self.config['experiment-params']
        self.input_dir = experiment_params['input_dir']
        self.train_dir = os.path.join(self.input_dir, "train")
        self.random_seed = int(experiment_params['random_seed'])
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        time_str = datetime.now().strftime('%H_%M_%S')
        self.out_dir = os.path.join(experiment_params['output_dir'],
                                    experiment_params['exp_name']+"_"+time_str)
        os.makedirs(self.out_dir, exist_ok=True)
        shutil.copy("client/config.yml", self.out_dir)
        logger.debug("Labeled Directory {} \n Output Dir {}".format(self.train_dir, self.out_dir))

        self.search_id = SearchId(value=str(uuid.uuid4()))
        signal.signal(signal.SIGINT, self.stop)

        self.positives = 0
        self.negatives = 0
        self.model_version = -1

        self.shuffle_and_generate_index_files()
        self.setup_search_config()
        self.add_train_test_data()

        # Stats Params
        stats_condition = "model"  # Model Version or Time Based
        self.stats_queue = queue.Queue()
        self.stats_stop_event = threading.Event()
        self.stats_interval = 1
        self.delphi_stats = get_stats(stats_condition,
                                      self.stubs,
                                      self.search_id,
                                      self.stats_stop_event,
                                      self.stats_queue,
                                      self.stats_interval)
        self.threads = [] # TODO

    def setup_search_config(self):
        supported_extractors = ["mobilenet_v2", "mpncov_resnet50", "resnet50"]

        retrain_config = self.config['retrain_policy']

        # TODO Remove Hardcoding
        retrain_policy = RetrainPolicyConfig(
                            percentage=PercentageThresholdPolicyConfig(
                                threshold=0.1,
                                onlyPositives=False,))
        strategies_config = self.config['train_strategy']

        train_strategies = []
        for strategy in strategies_config:
            model_config = strategy['model']
            condition_config = strategy['condition']
            assert model_config['feature_extractor'] in supported_extractors

            if model_config['type'] == "svm":
                mode = SVMMode.MASTER_ONLY
                if model_config['mode'] == 'distributed':
                    mode = SVMMode.DISTRIBUTED

                model = ModelConfig(
                    svm=SVMConfig(
                        mode=mode,
                        featureExtractor=model_config['feature_extractor'],
                        probability=model_config['probability'],
                        linearOnly=model_config['linear_only'],
                    )
                )

            if condition_config['type'] == "examples_per_label":
                train_strategies.append(
                    ModelConditionConfig(
                       examplesPerLabel=ExamplesPerLabelConditionConfig(
                           count=int(condition_config['count']),
                           model=model
                       )
                    )
                )

        param = self.config['selector']['topk']
        # TODO Remove Hardcoding
        reex = ReexaminationStrategyConfig(none=NoReexaminationStrategyConfig())
        has_init_examples = True #os.path.exists(os.path.join(self.config['root_dir'], "data", "train"))
        selector = SelectorConfig(topk=TopKSelectorConfig(k=param['k'], batchSize=param['batchSize'],
                        reexaminationStrategy=reex))

        def get_default_scopecookies():
            """
            Load and parse `$HOME/.diamond/NEWSCOPE`
            :return:
            """
            scope_file = os.path.join(os.environ['HOME'], '.diamond', 'NEWSCOPE')
            data = open(scope_file, 'rt').read()
            return [data]

        filters = []
        cookies = get_default_scopecookies()
        attributes = ['Device-Name', 'Display-Name', '_ObjectID', '_rows.int', '_cols.int'] + \
                     ['_filter.THUMBNAIL.heatmap.png', 'hyperfind.thumbnail-display'] + \
                     ['_filter.THUMBNAIL.patches', 'thumbnail.jpeg'] + \
                     [ATTR_FILTER_SCORE % (str(f)) for f in filters]


        logger.debug("Create Delphi Search {}".format(len(self.stubs)))
        # [stub.CreateSearch(search) for stub in self.stubs]
        len_stubs = len(self.stubs)
        for i, stub in enumerate(self.stubs[::-1]):
            node_index = len_stubs - (i + 1)
            dataset = Dataset(diamond=DiamondDataset(
                        filters=filters,
                        hosts=[self.hosts[node_index]],
                        cookies=cookies,
                        attributes=attributes
            ))
            search = CreateSearchRequest(
                        searchId=self.search_id,
                        nodes=self.nodes,
                        nodeIndex=node_index,
                        trainStrategy=train_strategies,
                        retrainPolicy=retrain_policy,
                        onlyUseBetterModels=False,
                        dataset=dataset,
                        selector=selector,
                        hasInitialExamples=True,
                        metadata="positive"
            )
            stub.CreateSearch(search)
            time.sleep(1)
        time.sleep(5)

    def shuffle_and_generate_index_files(self):
        """
        Reads stream.txt from input dir and shuffles and
        sends index files to the hosts
        """
        src_file = os.path.join(self.input_dir, "stream.txt")
        assert(os.path.exists(src_file))

        input_files = open(src_file).read().splitlines()

        random.Random(self.random_seed).shuffle(input_files)

        # divide the files among the host machines
        num_hosts = len(self.hosts)

        div_files = [input_files[i::num_hosts] for i in range(num_hosts)]

        dest_path = os.path.join(INDEX_PATH)
        filename = os.path.basename(INDEX_PATH)
        src_path = "/tmp/{}".format(filename)
        for i, host in enumerate(self.hosts):
            with open(src_path, "w") as f:
                f.write("\n".join(div_files[i]))
            # scp to hosts
            cmd = "scp {} root@{}:{}".format(src_path, host, dest_path)
            call(cmd.split(" "))
        return

    def add_train_test_data(self):
        """
        Sending train and test data to the master server
        """
        for data_type in ['test', 'train']:
            self.add_examples(self.stubs[0], self.to_examples(os.path.join(self.input_dir, data_type)), data_type)


    # def to_examples(self, input_data):
    #     if not os.path.exists(input_data):
    #          ogger.error("Path does not exist")
    #     image_root = self.config['root_dir']
    #      xamples = []
    #     seperator = ' '
    #     if os.path.isfile(input_data):
    #          ontents = open(input_data).read().splitlines()
    #         for content in contents:
    #              abel, path = content.split(seperator)
    #             examples.append((label, os.path.join(image_root, path)))
    #     else:
    #         raise NotImplementedError()

    #       return examples

    def to_examples(self, input_data):
        """
        Formatting entries in directory -> (label, path)
        """
        if not os.path.exists(input_data):
            logger.error("Path does not exist")
        examples = []
        for example_dir in Path(input_data).iterdir():
            label = example_dir.name
            for i, path in enumerate(example_dir.iterdir()):
                examples.append((label, path))

        return examples



    def add_examples(self, stub, examples, example_type):
        """
        Adding examples to labeled queue
        """
        if example_type.lower() == 'test':
            example_set = ExampleSetWrapper(value=ExampleSet.TEST)
        else:
            example_set = ExampleSetWrapper(value=ExampleSet.TRAIN)
        example_queue = queue.Queue()
        future = stub.AddLabeledExamples.future(to_iter(example_queue))
        example_queue.put(LabeledExampleRequest(searchId=self.search_id))
        for label, path in examples:
            if example_type == 'train':
                if label == '0':
                    self.negatives += 1
                else:
                    self.positives += 1

            example_queue.put(LabeledExampleRequest(example=LabeledExample(
                label=label,
                content=open(path, 'rb').read(),
                exampleSet=example_set,
            )))

        example_queue.put(None)
        future.result()

    def stats_logging(self):
        """
        Logs search status when model version changes
        Also downloads the model file
        """
        try:
            logger.info("Start logging")
            # logs = open(os.path.join(self.out_dir, "search-stats.json"), "w")
            count = 1
            while not self.stats_stop_event.wait(timeout=self.stats_interval):
                stats = self.stats_queue.get()
                logger.info("Stats {}".format(stats))
                if stats is None:
                    break
                if not stats:
                    continue
                if stats['version'] != self.model_version:
                    self.model_version = stats['version']
                    model_download = self.stubs[0].ExportModel(self.search_id)
                    logger.info("MODEL {} {}".format(model_download.version, model_download.trainExamples))
                    train_examples = model_download.trainExamples
                    for k, v in train_examples.items():
                        stats['train_{}'.format(k)] = int(v)
                    model = model_download.content
                    with open(os.path.join(self.out_dir, 'model-{}.zip'.format(self.model_version)), 'wb') as f:
                        f.write(model)

                with open(os.path.join(self.out_dir, "search-stats-{}.json".format(count)), "w") as f:
                    stats['positives'] = self.positives
                    stats['negatives'] = self.negatives
                    count += 1
                    json.dump(stats, f)
            logger.info("Stats Done !!")
        except Exception as e:
            logger.error(e)
            self.stop()
            raise e



    def start(self):
        self.index = 0
        self.tps = []
        self.fps = []
        self.fns = []
        self.images_processed = 0
        [stub.StartSearch(self.search_id) for stub in self.stubs]
        logger.info("create done starting search")
        self.model_version = -1
        self.global_start = time.time()
        # time.sleep(2)
        logger.info("Get searches")
        found = False
        while not found:
            # searches = self.stubs[0].GetSearches(Empty())
            # if len(searches):
            #     logger.info("Search found")
            #     break
            for search_info in self.stubs[0].GetSearches(Empty()):
                logger.info("Search {} {}".format(MessageToJson(search_info),
                            self.search_id.value == search_info.searchId.value))
                if self.search_id.value == search_info.searchId.value:
                    logger.info("Search found")
                    found = True
                    break
        threading.Thread(target=self.stats_logging).start()
        threading.Thread(target=self.delphi_stats.start).start()
        negative_key = "negative" # "/0/"
        results = [stub.GetResults(self.search_id) for stub in self.stubs]

        try:
            def _result_thread():
                node_assignments = cycle(range(len(self.stubs)))
                result_break = len(self.stubs)
                not_found = 0
                while True:
                    node_id = next(node_assignments)
                    start_time = time.time()
                    result = None
                    while result is None:
                        if time.time() - start_time > 1:
                            break
                        result = next(results[node_id])

                    if result is None:
                        not_found += 1
                        if not_found >= result_break:
                            break
                    else:
                        not_found = 0

                    attributes = result.attributes
                    device_name = STRING_CODEC.decode(attributes['Device-Name'])
                    obj_id = STRING_CODEC.decode(attributes['_ObjectID'])
                    label = '0' if negative_key in obj_id else '1'
                    if label == '0':
                        self.negatives += 1
                    else:
                        self.positives += 1
                    labeled = {obj_id: label}
                    logger.info("{} {}".format(device_name, labeled))
                    request = AddLabeledExampleIdsRequest(
                                searchId=self.search_id,
                                examples=labeled)
                    # TODO Add think time
                    self.stubs[node_id].AddLabeledExampleIds(request)
                logger.info("Result outside loop ?? {}".format(result is None))

                # for result in stub.GetResults(self.search_id):
                #     if result is None:
                #         break

                #     attributes = result.attributes
                #     device_name = STRING_CODEC.decode(attributes['Device-Name'])
                #     obj_id = STRING_CODEC.decode(attributes['_ObjectID'])
                #     label = '0' if negative_key in obj_id else '1'
                #     if label == '0':
                #         self.negatives += 1
                #     else:
                #         self.positives += 1
                #     labeled = {obj_id: label}
                #     logger.info("{} {}".format(device_name, labeled))
                #     request = AddLabeledExampleIdsRequest(
                #                 searchId=self.search_id,
                #                 examples=labeled)
                #     # TODO Add think time
                #     stub.AddLabeledExampleIds(request)
                # logger.info("Result outside loop ?? {}".format(result is None))

            # threading.Thread(target=_result_thread).start()
            _result_thread()
        except Exception as e:
            logger.error(e)
            self.stop()
            raise e


    def stop(self, *args):
        logger.info("Stop called")
        self.stats_stop_event.set()
        [stub.StopSearch(self.search_id) for stub in self.stubs]
        time.sleep(5)
        pid = os.getpid()
        logger.info("Killing")
        os.kill(pid, signal.SIGKILL)


def main():

    config_path = sys.argv[1] if len(sys.argv) > 1 \
                    else (Path.cwd() / 'client/config.yml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    client = DelphiClient(config)
    try:
        client.start()
    except (KeyboardInterrupt, Exception) as e:
        logger.error(e)
        client.stop()
        time.sleep(10)
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)


if __name__ == '__main__':
    main()