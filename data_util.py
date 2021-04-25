import music21
from tqdm import tqdm
import os



def collect_ABCFormat_data(abc):
    '''

    :param abc: an ABCFormat Object from music21
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
            dic['note'], dic['beat'], dic['beat_strength'] = note_list, [str(n.beat) for n in notes], [n.beatStrength
                                                                                                         for n in notes]
            output.append(dic)

    print('process completed! accept songs: {0}/{1}'.format(len(output), len(abc)))

    return (output)