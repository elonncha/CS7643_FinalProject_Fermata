from tqdm import tqdm

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