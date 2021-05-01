import numpy as np
import copy


def parse_folk_by_txt(meter = '4/4', seq_len_min = 256, seq_len_max = 256+32):
    '''
    parse original folk.txt into note, measure, and song_id
    :param meter: meter filter
    :param seq_len_min: seq_len filter
    :param seq_len_max: seq_len filter
    :return: note(nested list), measure(nested list), and song_id(list)
    '''
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
                song = raw[ite+2].replace('\n', '').split()
                note.append(song)
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
    :return: sliced sections (nested list)
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
    '''
    build the vocabulary of music notes
    :param list:
    :return: list of unique elements in the input
    '''
    u = np.array([])
    for i in list:
        u = np.concatenate((np.unique(np.array(i)), u))
    u = np.unique(u)
    return(u)



def manual_encoding(array, dic):
    '''
    encode list numerically by the orders of the dictionary
    :param array: array to be encoded
    :param dic: vocabulary dictionary
    :return: encoded array
    '''
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            id = np.argwhere(array[i,j] == dic)[0][0]
            array[i,j] = id

    return(array)



def load_data():
    '''

    :return: note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id
    '''
    # load original data
    note, measure, song_id = parse_folk_by_txt(meter='4/4', seq_len_min=256, seq_len_max=256 + 32)

    # slice by fractions
    note_past, note_target, note_future = slicing_by_fraction(note, past_fraction=0.3, future_fraction=0.3)
    measure_past, measure_mask, measure_future = slicing_by_fraction(measure, past_fraction=0.3, future_fraction=0.3)

    # add paddings
    note_past, note_target, note_future = add_padding(note_past, position='left'), add_padding(note_target, position='right'), add_padding(note_future, position='right')
    measure_past, measure_future = add_padding(measure_past, position='left'), add_padding(measure_future, position='right')

    for i in note_target:
        i.insert(0,'<s>') #extra padding for note_target

    # type change
    note_past,note_target, note_future, \
    measure_past, measure_future = np.array(note_past),  np.array(note_target), np.array(note_future), \
                                   np.array(measure_past), np.array(measure_future)

    # build note dictionary
    note_dic = np.unique(np.concatenate((np.unique(note_past), np.unique(note_future), build_dictionary(note_target))))
    note_dic_count = note_dic.shape[0]

    # encode measure
    measure_past[measure_past == '<e>'] = -3
    measure_past[measure_past == '<s>'] = -2
    measure_future[measure_future == '<e>'] = -3
    measure_future[measure_future == '</s>'] = -1
    measure_past = np.array(measure_past, dtype='int')
    measure_future = np.array(measure_future, dtype='int')

    # encode note
    note_past = manual_encoding(note_past, note_dic)
    note_target = manual_encoding(note_target, note_dic)
    note_future = manual_encoding(note_future, note_dic)
    note_past = np.array(note_past, dtype='int')
    note_target = np.array(note_target, dtype='int')
    note_future = np.array(note_future, dtype='int')

    return(note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id)



def note_decoder(note, note_dic):
    '''
    decode numerical representation to original abc notes
    :param note:
    :param note_dic:
    :return:
    '''
    note_decoded = []
    for s in note:
        decoded = note_dic[s]
        mask = np.argwhere(np.isin(decoded, ['<e>', '<s>', '</s>']) == False).flatten()
        decoded = decoded[mask]
        note_decoded.append(decoded.tolist())

    return(note_decoded)



def reverse_note_to_abc(note, measure):
    '''
    reverse a list of notes into a character string
    :param note:
    :param measure:
    :return:
    '''
    str = ""
    current_measure = 0
    for step in range(len(measure)):
        if measure[step] == current_measure:
            if step == len(measure) - 1:
                str += note[step] + ' |\n'
            else:
                str += note[step] + ' '
        else:
            current_measure += 1
            str += '| ' + note[step] + ' '
    return(str)



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
    measure_past_val = measure_past_val.tolist()
    measure_past_val = [[n for n in s if n>=0] for s in measure_past_val]
    measure_future_val = measure_future_val.tolist()
    measure_future_val = [[n for n in s if n >= 0] for s in measure_future_val]
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