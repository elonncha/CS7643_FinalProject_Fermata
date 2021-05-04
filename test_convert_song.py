import pickle
from data_util import *

data = INPAINT(data_root='./dataset_split', ds_type='test')
note_dic = data.get_note_dict()

results_root = './results_01/Seq2Seq_LSTM/prediction'
with open(results_root, 'rb') as pickle_r:
    dict = pickle.load(pickle_r, encoding='bytes')
    raw_score = dict[b'raw_score']
    note_target_pred = dict[b'abc']

#i know the code is ugly
songs = []
ground_truth = []
with open('./dataset_split/test', 'rb') as pickle_r:
    dict = pickle.load(pickle_r, encoding='bytes')
    past = note_decoder(dict[b'note_past'], note_dic)
    future = note_decoder(dict[b'note_future'], note_dic)
    target = note_decoder(dict[b'target'], note_dic)
    predictions = note_decoder(np.argmax(raw_score, axis=2), note_dic)
    for i in enumerate(note_target_pred):
        # idk why the predictoin was a tuple
        y = i[1]
        for idx, j in enumerate(y):
            x = np.delete(j[1:], np.argwhere(j[1:]=='<s>'))
            x = np.delete(x, np.argwhere(x=='<e>'))
            x = np.delete(x, np.argwhere(x=='</s>'))
            song = np.concatenate((np.delete(past[idx], np.argwhere(past[idx]=='<e>')), x, np.delete(future[idx], np.argwhere(past[idx]=='<e>'))))
            actual = np.concatenate((np.delete(past[idx], np.argwhere(past[idx]=='<e>')), target[idx], np.delete(future[idx], np.argwhere(past[idx]=='<e>'))))
            songs.append(song)
            ground_truth.append(actual)

# test
i=130
print(songs[i])
print(ground_truth[i])
print((target[i]))
print((predictions[i]))