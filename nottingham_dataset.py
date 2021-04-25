from dataset import MusicDataset
from data_util import *
import music21
import os
import pandas as pd

class NottinghamDataset(MusicDataset):
    def __init__(self,dataset_name = 'nottingham'):
        super(NottinghamDataset, self).__init__(dataset_name = dataset_name)

    def parse_abc(self, abc_file):
        '''
        parse the raw abc file (txt) and return a list of songs with selected features (notes, beats, etc,.)
        :param abc_file: name of the file to be parsed, or 'all' if parsing all files in the directory. e.g. 'jigs.abc'
        :return: output, a python list. At the same time write a csv into data/cache/raw
        '''
        self.data_list = None
        output = []

        # parse abc using music21.converter.parse
        if abc_file != 'all':
            path = 'data/' + self.dataset_name + '/' + abc_file
            print('Parsing {0}...'.format(abc_file))
            abc = music21.converter.parse(path)
            print('Parse succeeds!')

            output += collect_ABCFormat_data(abc)

        else:
            file_name = os.listdir('data/' + self.dataset_name)
            for f in file_name:
                path = 'data/' + self.dataset_name + '/' + f
                print('Parsing {0}...'.format(f))
                abc = music21.converter.parse(path)
                print('Parse succeeds!')

                output += collect_ABCFormat_data(abc)

        self.data_list = output

        df = pd.DataFrame(output)
        df.song_id = range(len(output))
        df.set_index('song_id')
        df.to_csv('data/cache/raw/{0}.csv'.format(self.dataset_name))

        return(output)


    def tokenize(self):
        pass




ds = NottinghamDataset()
outs = ds.parse_abc('slip.abc')

