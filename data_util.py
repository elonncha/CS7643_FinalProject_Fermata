import music21
from tqdm import tqdm
import numpy as np
import os
import torch
from sklearn import preprocessing
import pandas as pd

def collect_ABCFormat_data(abc):
    '''

    :param abc: an ABCFormat Object from music21 (nottingham dataset)
    :return: a list of songs with notes, beats, and beatStrength
    '''
    output = []
    print('process songs...')

    for song_idx, song in enumerate(tqdm(abc)):

        notes = song.flat.notes
        dic = {'song_id': song_idx, 'note': None, 'beat': None, 'beat_strength': None}

        accept_song = True
        note_list = []
        for n_idx, n in enumerate(notes):
            try:
                note_list.append(n.nameWithOctave)  # only Note has property nameWithOctave
            except:
                if len(n.notes) == 0:  # empty notes, fill na with previous notes
                    accept_song = False
                    break
                else:
                    note_list.append(n.notes[0].nameWithOctave)  # for Chord Symbol, we use the first note

        if accept_song:
            dic['note'], dic['beat'], dic['beat_strength'] = note_list, [str(n.beat) for n in notes], [n.beatStrength for n in notes]
            output.append(dic)

    print('process completed! accept songs: {0}/{1}'.format(len(output), len(abc)))

    return (output)



def collect_ABCFormat_data_single(abc):
    '''

    :param abc: an ABCFormat Object with SINGLE song from music21 (folk dataset)
    :return: a list of songs with notes, beats, and beatStrength
    '''

    notes = abc.flat.notes
    dic = {'note': [],
            'beat': [],
            'beat_strength': []}

    for n in notes:
        try:
            dic['note'].append(n.nameWithOctave)
            dic['beat'].append(str(n.beat))
            dic['beat_strength'].append(n.beatStrength)
        except:
            dic = {'note': [],
                   'beat': [],
                   'beat_strength': []} # bad sample, return empty
            break

    return (dic)



def parse_folk_by_txt(meter = '4/4', seq_len_min = 256, seq_len_max = 256+32):
    file_name = 'data/folk.txt'
    with open(file_name, 'r') as f:
        raw = f.readlines()
        f.close()

    raw_txt, note, length = [], [], []
    for ite, line in enumerate(raw):
        if "T:" in line:
            continue
        elif 'M:' in line:
            if meter in line:
                # raw = raw[ite-1] + line + raw[ite+1] + raw[ite+2]
                song = raw[ite+2].replace('\n', '').split()

                note.append(song)
                # raw_txt.append(raw)
        else:
            continue


    song_id = [idx for idx, n in enumerate(note) if seq_len_min<= len(n)-n.count('|') <= seq_len_max]
    note = [note[i] for i in song_id]


    measure = []
    for i in range(len(note)):

        current_m = 0
        measure_embed = []
        for n in note[i]:
            if n == '|':
                current_m += 1
            else:
                measure_embed.append(current_m)

        measure.append(measure_embed)
        note[i] = [x for x in note[i] if x != '|']


    return(note, measure, song_id)



def slicing_by_fraction(list, past_fraction = 0.3, future_fraction = 0.3):
    '''
    :param list: list to be sliced
    :param past_fraction: percent
    :param future_fraction: percent
    :return: sliced sections
    '''
    total_length = np.array([len(x) for x in list])
    past_length = np.floor(past_fraction * total_length)
    mask_length = np.floor((1 - past_fraction - future_fraction) * total_length)

    past = [n[0:int(past_length[idx])] for idx, n in enumerate(list)]
    mask = [n[int(past_length[idx]): int(past_length[idx] + mask_length[idx])] for idx, n in enumerate(list)]
    future = [n[int(past_length[idx] + mask_length[idx]):] for idx, n in enumerate(list)]

    return(past, mask, future)



def add_padding(list, position):
    '''
    add <e> to the end of each sequence to make the sequence length consistent by paddings
    :param list: a list of sequences (list) of different seq_length
    :param position: where to pad: ['left', 'right']
    :return: a list of sequences (list) of consistent seq_length

    '''
    max_length = max(len(x) for x in list)
    length_diff = np.array([max_length - len(x) for x in list])

    if position == 'left': # past context
        padded_list = [ ['<e>'] * int(length_diff[idx]) + ['<s>'] + x for idx, x in enumerate(list)]

    else: # position == 'right', future context
        padded_list = [x + ['</s>'] + ['<e>'] * int(length_diff[idx]) for idx, x in enumerate(list)]

    return(padded_list)



def build_dictionary(list):
    u = np.array([])
    for i in list:
        u = np.concatenate((np.unique(np.array(i)), u))
    u = np.unique(u)
    return(u)



def manual_encoding(array, dic):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            id = np.argwhere(array[i,j] == dic)[0][0]
            array[i,j] = id

    return(array)



def load_data():
    '''

    :return:
    '''
    # load original data
    note, measure, song_id = parse_folk_by_txt(meter='4/4', seq_len_min=256, seq_len_max=256 + 32)

    # slice by fractions
    note_past, note_target, note_future = slicing_by_fraction(note, past_fraction=0.3, future_fraction=0.3)
    measure_past, measure_mask, measure_future = slicing_by_fraction(measure, past_fraction=0.3, future_fraction=0.3)

    # add paddings
    note_past, note_future = add_padding(note_past, position='left'), \
                             add_padding(note_future, position='right')
    measure_past, measure_future = add_padding(measure_past, position='left'), \
                                   add_padding(measure_future, position='right')

    # type change
    note_past, note_future, measure_past, measure_future = np.array(note_past), \
                                                            np.array(note_future), \
                                                            np.array(measure_past), \
                                                            np.array(measure_future)

    # build note dictionary
    note_dic = np.unique(np.concatenate((np.unique(note_past), np.unique(note_future), build_dictionary(note_target))))
    note_char_total_count = note_dic.shape[0]

    # encode measure
    measure_past[measure_past == '<e>'] = -3
    measure_past[measure_past == '<s>'] = -2
    measure_future[measure_future == '<e>'] = -3
    measure_future[measure_future == '</s>'] = -1
    measure_past = np.array(measure_past, dtype='int')
    measure_future = np.array(measure_future, dtype='int')

    # encode note
    note_past = manual_encoding(note_past, note_dic)
    note_future = manual_encoding(note_future, note_dic)
    note_past = np.array(note_past, dtype='int')
    note_future = np.array(note_future, dtype='int')

    for i in range(len(note_target)):
        for j in range(len(note_target[i])):
            note_target[i][j] = np.argwhere(note_dic == note_target[i][j])[0][0]


    return(note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id)



def note_decoder(note, note_dic):
    note_decoded = []
    for s in note:
        decoded = note_dic[s]
        mask = np.argwhere(np.isin(decoded, ['<e>', '<s>', '</s>']) == False).flatten()
        decoded = decoded[mask]
        note_decoded.append(decoded.tolist())

    return(note_decoded)



def reverse_note_to_abc(note, measure):
    #TODO


def reconstruct_song(song_id_val,
                     note_past_val, note_future_val, note_target_predicted, note_dic,
                     measure_past_val):

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
    # TODO


    # reconstruct new songs
    note_predicted = []
    for i in range(len(song_id_val)):
        new_song = note_past_val[i] + note_target_predicted[i] + note_future_val[i]
        note_predicted.append(new_song)


    # write into raw txt format
    song_predicted_raw = []

    for i in range(len(song_raw_val)):
        # TODO
        pass