from data_util import *
import torch
import numpy as np
from sklearn import preprocessing


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
# dataset split





