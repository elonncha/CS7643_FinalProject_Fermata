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




    def load_dataset(self): # check if you have nottingham.csv file under data/cache/raw

        # load raw from cache , add start and end tokens and paddings
        df = pd.read_csv('data/cache/raw/nottingham.csv')

        note, beat, beatStrength = [['<s>'] + ast.literal_eval(n) + ['</s>'] for n in df['note']], \
                                   [['<s>'] + ast.literal_eval(n) + ['</s>'] for n in df['beat']], \
                                   [['<s>'] + ast.literal_eval(n) + ['</s>'] for n in df['beat_strength']]

        self.note, self.beat, self.beatStrength = note, beat, beatStrength


    def tokenize(self):

        pass



ds = FolkDataset()
ds.parse_abc()

