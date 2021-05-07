class RetrieverStats(object):

    def __init__(self, dictionary):
        assert 'total_objects' in dictionary

        self.dropped_objects = 0
        self.false_negatives = 0

        for key in dictionary:
            setattr(self, key, dictionary[key])
