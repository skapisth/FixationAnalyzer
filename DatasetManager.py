from collections import deque
from functools import partial
import fixationanalyzer as fa
from fixationanalyzer import Participant
from fixationanalyzer import FeatureExtractor
from itertools import chain
from math import floor
import numpy as np
import csv


class DatasetManager(object):
    """
    Must Modify To Verify Dataset Integrity
    Verify that all participants see all paintings

    """
    def __init__(self,
                    fixation_filename,
                    training_fraction=.7,
                    num_chunks=10,
                    classification_type='exp_v_nov',
                    ):
        self.fixation_filename = fixation_filename
        self.training_fraction = training_fraction
        self.num_chunks = num_chunks
        self.classification_type = classification_type


        self.participants = self.__process_input(self.fixation_filename,self.num_chunks)
        self.data_chunks, self.label_chunks = self.__chunk_dataset(self.participants)
        self.num_rotations = 0


    def get_train_test(self):
        split_index = floor(self.training_fraction * self.num_chunks)

        train_participants = chain(*(self.data_chunks[i] for i in range(split_index)))
        test_participants = chain(*(self.data_chunks[i] for i in range(split_index,self.num_chunks)))

        train_labels = list( chain( *(self.label_chunks[i] for i in range(split_index) ) ) )
        test_labels = list( chain( *(self.label_chunks[i] for i in range(split_index,self.num_chunks) ) ) )

        train_data = [p.fixation_data for p in train_participants]
        test_data = [p.fixation_data for p in test_participants]


        return train_data,train_labels, test_data, test_labels

    def rotate(self):
        
        if self.num_rotations > self.num_chunks:
            raise RuntimeError("You can't rotate your dataset more times ({0}) than there are chunks ({1})!.".format(self.num_rotations,self.num_chunks))
        self.data_chunks.rotate()
        self.label_chunks.rotate()
        self.num_rotations += 1


    def __chunk_dataset(self,participants):
        if self.classification_type == 'exp_v_nov': #exp vs nov
            def _assessment_func(participant):
                return participant.expert

        elif self.classification_type == 'inst_v_ninst': #inst vs Non
            def _assessment_func(participant):
                return participant.instructed

        elif self.classification_type == 'four': #4-way
            def _assessment_func(participant):
                truth_table = [[0,1],
                                [2,3]]
                return truth_table[participant.expert][participant.instructed]

        label_separated = [[],[]] if self.classification_type != 'four' else [[],[],[],[]]
        for p in participants:
            lbl = _assessment_func(p)
            label_separated[lbl].append(p)

        data_chunks = [ [] for i in range(self.num_chunks) ]
        label_chunks = [ [] for i in range(self.num_chunks) ]

        criteria = any( len(l) for l in label_separated )
        chunk_index = 0
        while criteria:
            slice = []
            slice_labels = []
            for lbl in range( len(label_separated) ):
                if len(label_separated[lbl]) == 0:
                    continue

                slice.append(label_separated[lbl].pop(0))
                slice_labels.append(lbl)

            data_chunks[chunk_index].extend(slice)
            label_chunks[chunk_index].extend(slice_labels)

            chunk_index = (chunk_index + 1) % self.num_chunks
            criteria = any( len(l) for l in label_separated )

        #END WHILE LOOP
        data_chunks = deque(data_chunks)
        label_chunks = deque(label_chunks)

        if any(chunk==[] for chunk in data_chunks):
            raise RuntimeError("All your folds are not populated with data. Can you get more data?.")

        return data_chunks,label_chunks

    def __process_input(self,fixation_filename,num_data_chunks):

        participants = {}
        #parsing the csv file
        with open(fixation_filename,'r') as f:
            reader = csv.reader(f)
            #retreiving the column headers from the csv file
            column_headers = next(reader)
            column_headers = [header.encode('ascii',errors='ignore').decode('utf-8') for header in column_headers]
            for row in reader:
                #creating a dictionary for each line of data
                row_dict = dict( zip(column_headers,row) )
                subject = row_dict['subject-id']
                #updating list of participant data
                if subject in participants:
                    participants[subject].append(row_dict)
                else:
                    participants[subject] = [row_dict]


        participants = [Participant(subject_id,data) for subject_id,data in participants.items()]
        return participants
