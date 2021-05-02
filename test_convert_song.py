import pickle
from data_util import *

# temporarily copied here to work on it before putting back in data_util
def reconstruct_song(song_id_val,
                     note_past_val, note_future_val, note_target_predicted, note_dic,
                     measure_past_val, measure_future_val, measure_mask_val):

    # load raw txt song data
    file_name = 'data/folk.txt'
    with open(file_name, 'r') as f:
        raw = f.readlines()
        f.close()
    song_raw = []
    for ite, line in enumerate(raw):
        if 'M:4/4' in line:
            song = [raw[ite-1], line, raw[ite+1], raw[ite+2]]
            song_raw.append(song)
        else:
            continue
    song_raw_val = [song_raw[idx] for idx in song_id_val]

    # decode predictions
    note_past_val, note_target_predicted, note_future_val = note_decoder(note_past_val, note_dic), \
                                                            note_decoder(note_target_predicted, note_dic), \
                                                            note_decoder(note_future_val, note_dic)
    # decode measure
    # measure_past_val = measure_past_val.tolist()
    measure_past_val = [[n for n in s if n>=0] for s in measure_past_val]
    # measure_future_val = measure_future_val.tolist()
    measure_future_val = [[n for n in s if n >= 0] for s in measure_future_val]
    print('measure: ', measure_past_val, measure_future_val)
    measure_val = [measure_past_val[i] + measure_mask_val[i] + measure_future_val[i] for i in range(len(song_id_val))]

    # reconstruct new songs
    note_predicted = []
    for i in range(len(song_id_val)):
        new_song = note_past_val[i] + note_target_predicted[i] + note_future_val[i]
        note_predicted.append(new_song)

    # write into raw txt format
    output = copy.deepcopy(song_raw_val)
    for i in range(len(song_raw_val)):
        output[i][-1] = reverse_note_to_abc(note_predicted[i], measure_val[i])

    return(output)



results_root = './results/Seq2Seq_LSTM/prediction'

with open(results_root, 'rb') as pickle_r:
    dict = pickle.load(pickle_r, encoding='bytes')
    note_target_pred = dict[b'pred'].tolist()
    print('Predicted (Test): ', note_target_pred)

# did not shuffle data during training so works
data = INPAINT(data_root='./dataset_split', ds_type='test')

note_dic = data.get_note_dict()

note_past, measure_past, \
    note_future, measure_future, \
        target, measure_mask, song_id = data.note_past, data.measure_past, \
                                            data.note_future,data.measure_future, \
                                                data.target, data.measure_mask, data.song_id

note_past, measure_past, \
    note_future, measure_future = [d.tolist() for d in [note_past, measure_past, note_future, measure_future]]    


output = reconstruct_song(song_id, note_past, note_future, note_target_pred, note_dic, measure_past, measure_future, measure_mask)
# have to manually remove the padding first
print(output)
