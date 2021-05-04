import io
import torch
import csv
import ast
import math

def csv_to_tensor(filepath, max_song_len = 256):
    # read csv to list data
    data_all = []
    with open(filepath) as f:
        reader = csv.reader(f)
        data = list(reader)
        data_all.append(data)

    # clean data so it only contains the music
    data_all = data_all[0][1:]

    # convert data to clean list of lists, ~20 seconds
    N = len(data_all)

    data_clean = []
    for n in range(N):
        song_data = []
        for i in range(4):
            cell = ast.literal_eval(data_all[n][i])
            song_data.append(cell)
        data_clean.append(song_data)

    # get set of note names
    notenames_set = set() # set of notenames
    beat_lens_set = set() # set of beat lengths
    beat_strg_set = set() # set of beat strengths

    for i in range(len(data_clean)):
        notenames_set.update(set(data_clean[i][1]))
        beat_lens_set.update(set(data_clean[i][2]))
        beat_strg_set.update(set(data_clean[i][3]))
    notenames_set = sorted(list(notenames_set))
    beat_lens_set = sorted(list(beat_lens_set))
    beat_strg_set = sorted(list(beat_strg_set))

    # list of list of entries for notes, beat lengths, beat strengths, where the item in each
    # list corresponds to each song
    # data_clean = data_clean[:5] # uncomment to use less rows

    notes = [data_clean[i][1] for i in range(len(data_clean))]
    beat_lens = [data_clean[i][2] for i in range(len(data_clean))]
    beat_strg = [data_clean[i][3] for i in range(len(data_clean))]

    # convert to numbers / tensor
    # cuts to max_song_len if too long
    # create notes lists, encoded with notename_set
    notes_enc = []
    for song in notes:
        out_i = []
        for i, c in enumerate(song):
            out_i.append(notenames_set.index(c))
        out_i = out_i[:max_song_len]
        notes_enc.append(out_i)

    # create beat length lists, encoded with beat_lens_set
    beats_enc = []
    for song in beat_lens:
        beats_i = []
        for i, c in enumerate(song):
            beats_i.append(beat_lens_set.index(c))
        beats_i = beats_i[:max_song_len]
        beats_enc.append(beats_i)
        
    # rename beat strength as encoded list, float form probably good enough
    beat_strg_enc = []
    for song in beat_strg:
        beat_strg_i = []
        for i, c in enumerate(song):
            beat_strg_i.append(beat_strg_set.index(c))
        beat_strg_i = beat_strg_i[:max_song_len]
        beat_strg_enc.append(beat_strg_i)
    # now we have notes_enc, beats_enc, beat_strg_enc!

    # convert to tensors, pad to common dim
    # max_song_len = max(len(x) for x in notes_enc)

    notes_tensor = []
    beats_tensor = []
    beat_strg_tensor = []

    for i in range(len(notes_enc)):
        zeros = torch.zeros([max_song_len - len(notes_enc[i])]).tolist()
        
        notes_tensor.append(notes_enc[i] + zeros)
        beats_tensor.append(beats_enc[i] + zeros)
        beat_strg_tensor.append(beat_strg_enc[i] + zeros)

    # convert to tensors
    notes_tensor = torch.Tensor(notes_tensor)
    beats_tensor = torch.Tensor(beats_tensor)
    beat_strg_tensor = torch.Tensor(beat_strg_tensor)

    # combine to three-channel tensor
    data_tensor = torch.stack((notes_tensor, beats_tensor, beat_strg_tensor), dim = 1).long()
    # now we have a tensor with the data - channel 0 is the note sequence, channel 1 
    # is the beat durations and channel 2 are the beat strengths. 

    # return the sets for decoding.
    return data_tensor, notenames_set, beat_lens_set, beat_strg_set

def train_val_test_split(data, train = .6, val = .2, test = .2):
    N = data.size(0)
    train_qty = math.floor(N*train)
    val_qty = math.floor(N*val)
    test_qty = math.floor(N*test)

    train_set = data[:train_qty, ...]
    val_set = data[train_qty:train_qty+val_qty, ...]
    test_set = data[train_qty+val_qty:train_qty+val_qty+test_qty, ...]

    return train_set, val_set, test_set


def batch_data(data, channel, batchsize):
    # this takes the songs, stretches them out to one long song, and converts them to 
    # num_batches columns.
    # take in data from csv_to_tensor
    # select the channel
    data = data[:, channel, :]
    # need to stretch out all songs to one long tensor.
    data = data.flatten()

    num_batches = data.size(0) // batchsize
    data = data.narrow(0,0, num_batches * batchsize) # trim data
    data = data.view(batchsize, -1).t().contiguous()
    return data

# this function is for the txt version, already flattened. 
def batch_data_txt(data, batchsize):
    # this takes the songs, stretches them out to one long song, and converts them to 
    # num_batches columns.
    # take in data from csv_to_tensor
    # select the channel
    # data = data[:, channel, :]
    # need to stretch out all songs to one long tensor.
    # data = data.flatten()

    num_batches = data.size(0) // batchsize
    data = data.narrow(0,0, num_batches * batchsize) # trim data
    data = data.view(batchsize, -1).t().contiguous()
    return data