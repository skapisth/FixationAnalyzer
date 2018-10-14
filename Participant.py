import numpy as np
from collections import OrderedDict
class Participant(object):
    def __init__(self,
                    subject_id,
                    row_data):
        # import pdb;pdb.set_trace()
        self.name = row_data[0]['subject']
        self.name_id  = int(subject_id)
        self.expert = bool(int(row_data[0]['exp']))
        self.instructed = bool(int(row_data[0]['inst']))

        self.check_row_data = row_data
        self.fixation_data = self.__parse_fixation_data(row_data)
        self.paintings_seen = list( self.fixation_data.keys() )
        #self.features = self.__generate_features(self.fixation_data)

    # def __generate_features(self,fixation_data):
    #     pass

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __parse_fixation_data(self,row_data):
        paintings = {}
        for row in row_data:
            painting_id = int(row['stimulus-id'])
            fixation_dict = {'X':int(round(float(row['positionX']),0)),
                             'Y':int(round(float(row['positionY']),0)),
                             'dur':float(row['dur'])}
            if painting_id in paintings:
                paintings[painting_id].append(fixation_dict)
            else:
                paintings[painting_id] = [fixation_dict]

        sorted_painting_ids = sorted( paintings.keys() )
        sorted_paintings = OrderedDict()
        for pid in sorted_painting_ids:
            sorted_paintings[pid] = paintings[pid]

        return sorted_paintings
