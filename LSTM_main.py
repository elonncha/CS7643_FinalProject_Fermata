from data_util import *
import torch
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

# load original data
note, measure, song_id = parse_folk_by_txt(meter = '4/4', seq_len_min = 256, seq_len_max = 256+32)
# slice by fractions
note_past, note_target, note_future = slicing_by_fraction(note, past_fraction = 0.3, future_fraction = 0.3)
measure_past, measure_mask, measure_future = slicing_by_fraction(measure, past_fraction = 0.3, future_fraction = 0.3)
# add paddings
note_past, note_future = add_padding(note_past, position = 'left'), \
                         add_padding(note_future, position = 'right')
measure_past, measure_future = add_padding(measure_past, position = 'left'), \
                               add_padding(measure_future, position = 'right')
# type change
note_past, note_future, measure_past, measure_future = np.array(note_past), np.array(note_future), np.array(measure_past), np.array(measure_future)



# build note dictionary
note_dic = np.unique(np.concatenate((np.unique(note_past), np.unique(note_future), build_dictionary(note_target))))
note_char_total_count = note_dic.shape[0]

# encode measure
measure_past[measure_past == '<e>'] = -3
measure_past[measure_past == '<s>'] = -2
measure_future[measure_future == '<e>'] = -3
measure_future[measure_future == '</s>'] = -1
measure_past = np.array(measure_past, dtype = 'int')
measure_future = np.array(measure_future, dtype = 'int')

num_measure = 3 + 1 + np.max(measure_future)

# encode note
note_past = manual_encoding(note_past, note_dic)
note_future = manual_encoding(note_future, note_dic)
note_past = np.array(note_past, dtype = 'int')
note_future = np.array(note_future, dtype = 'int')


note_past, note_future, measure_past, measure_future = torch.from_numpy(note_past),\
                                                       torch.from_numpy(note_future),\
                                                       torch.from_numpy(measure_past),\
                                                       torch.from_numpy(measure_future)


# get dataset split idx
np.random.seed(0)
train_percentage = 0.70
rand_idx = np.random.randint(0, note_past.shape[0], note_past.shape[0])
train_size = int(train_percentage * note_past.shape[0])

train_idx = rand_idx[:train_size]
val_idx = rand_idx[train_size:train_size+int((rand_idx.shape[0]-train_size)/2)]
test_idx = rand_idx[int((rand_idx.shape[0]-train_size)/2):]
ds_idxs = [train_idx, val_idx, test_idx]
ds_name = ['train', 'val', 'test']

# split dataset
data_r = [note_past, note_future, measure_past, measure_future]
for i, idx in enumerate(ds_idxs):
    path = './dataset_split/'+ds_name[i]
    with open(path, 'wb') as pickle_w:
        note_past, note_future, measure_past, measure_future = [x[idx].numpy() for x in data_r] # for this temporarily converted tensor back to numpy
        write = {b'note_past': note_past,
                b'note_future': note_future,
                b'measure_past': measure_past,
                b'measure_future': measure_future} 
        pickle.dump(write, pickle_w)
    # open test
    with open(path, 'rb') as pickle_r:
        dict = pickle.load(pickle_r, encoding='bytes')
        print(ds_name[i], dict[b'note_past'].shape)