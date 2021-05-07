import os
import torch

from google.protobuf.any_pb2 import Any

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model_trainer import TrainingStyle, DataRequirement
from delphi.model_trainer_base import ModelTrainerBase


class FinetuneTrainerBase(ModelTrainerBase):

    def __init__(self, context: ModelTrainerContext, train_start: int = 50):
        super().__init__()
        """
        Finetune model using shared positives and negatives
        available in server
        """
        self.context = context
        self.train_start = train_start

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def data_requirement(self) -> DataRequirement:
        return DataRequirement.DISTRIBUTED_FULL
        # if self.is_svm:
        #     return DataRequirement.DISTRIBUTED_FULL
        # else:
        #     return DataRequirement.DISTRIBUTED_POSITIVES

    @property
    def training_style(self) -> TrainingStyle:
        return TrainingStyle.DISTRIBUTED  # Independent training or model averaging ??

    @property
    def should_sync_model(self) -> bool:
        return False

    def message_internal(self, request: Any) -> Any:
        pass

