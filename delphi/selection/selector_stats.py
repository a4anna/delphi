from typing import Optional


class SelectorStats(object):

    def __init__(self, dictionary):
        assert 'processed_objects' in dictionary
        self.dropped_objects = 0
        self.passed_objects = None
        self.false_negatives = 0

        for key in dictionary:
            setattr(self, key, dictionary[key])

