from dataset import MusicDataset
from data_util import *
import music21
import os
import pandas as pd
import numpy as np
import ast

class FolkDataset(MusicDataset):
    def __init__(self,dataset_name = 'folk'):
        super(FolkDataset, self).__init__(dataset_name = dataset_name)

    def parse_abc(self):  # this function takes very long time to run
        '''
        parse the raw abc file (txt) and return a list of songs with selected features (notes, beats, etc,.)
        :return: output, a python list. At the same time write a csv into data/cache/raw
        '''
        self.data_list = None
        output = []

        # parse abc using music21.converter.parse
        file_name = os.listdir('data/' + self.dataset_name)
        for f in tqdm(file_name):
            try:
                path = 'data/' + self.dataset_name + '/' + f
                abc = music21.converter.parse(path)
                output.append(collect_ABCFormat_data_single(abc))
            except:
                continue

        print('process completed!')

        # write to csv
        df = pd.DataFrame(output)
        df.to_csv('data/cache/raw/{0}.csv'.format(self.dataset_name))

        return(output)


    def load_dataset(self, seq_len_lower = 128, seq_len_upper = 256): # check if you have folk.csv file under data/cache/raw
        '''

        :param seq_len_lower: lower bound of seq_len
        :param seq_len_upper: upper bound of seq_len
        :return: None
        '''

        # load raw from cache , add start and end tokens and paddings
        df = pd.read_csv('data/cache/raw/folk.csv')

        note, beat, beatStrength = [ast.literal_eval(n) for n in df['note']], \
                                   [ast.literal_eval(n) for n in df['beat']], \
                                   [ast.literal_eval(n) for n in df['beat_strength']]

        # only retain samples with desired seq_length
        note_portion, beat_portion, beatStrength_portion = [n for n in note if len(n) >= seq_len_lower and len(n) <= seq_len_upper], \
                                                           [n for n in beat if len(n) >= seq_len_lower and len(n) <= seq_len_upper], \
                                                           [n for n in beatStrength if len(n) >= seq_len_lower and len(n) <= seq_len_upper]

        self.note, self.beat, self.beatStrength = note_portion, beat_portion, beatStrength_portion
        return(note_portion, beat_portion, beatStrength_portion)



    def tokenize(self):

        pass



ds = FolkDataset()
ds.load_dataset()

past, mask, future = slicing_by_fraction(ds.note)


padded_past = add_padding(past, position = 'left')
padded_future = add_padding(future, position = 'right')



max(len(x) for x in padded_past) - min(len(x) for x in padded_past)
max(len(x) for x in padded_future) - min(len(x) for x in padded_future)