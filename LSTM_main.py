from data_util import *
import torch
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

# data load
note_past, note_target, note_future, \
measure_past, measure_mask, measure_future, \
note_dic = load_data()

# train-test split
np.random.seed(1)
note_past_train, note_past_test, \
note_future_train, note_future_test, \
note_target_train, note_target_test, \
measure_past_train, measure_pass_test, \
measure_future_train, measure_future_test = train_test_split(note_past,
                                                             note_future,
                                                             note_target,
                                                             measure_past,
                                                             measure_future, train_size = 0.8)

# test-validation split
note_past_val, note_past_test, \
note_future_val, note_future_test, \
note_target_val, note_target_test, \
measure_past_val, measure_pass_test, \
measure_future_val, measure_future_test = train_test_split(note_past_test,
                                                           note_future_test,
                                                           note_target_test,
                                                           measure_pass_test,
                                                           measure_future_test, test_size = 0.5)





'''
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
'''