from typing import Optional

from delphi.attribute_provider import AttributeProvider


class ResultProvider(object):

    def __init__(self, id: str, label: str, score: float, model_version: Optional[int],
            attributes: AttributeProvider, gt: bool, device: str = ""):
        self.id = id
        self.label = label
        self.score = score
        self.model_version = model_version
        self.attributes = attributes
        self.gt = gt
        self.device = device
