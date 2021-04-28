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





def to_tensor(padded_list):
    seq_array = np.array(padded_list)
    shape = seq_array.shape

    encoder = preprocessing.LabelEncoder()
    targets = encoder.fit_transform(seq_array.reshape(1,-1)[0,:]).reshape(shape[0], shape[1])

    seq_tensor = torch.from_numpy(targets)



