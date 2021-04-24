from dataset import MusicDataset
import music21

class NottinghamDataset(MusicDataset):
    def __init__(self,dataset_name = 'nottingham'):
        super(NottinghamDataset, self).__init__(dataset_name = dataset_name)

    def parse_abc_txt(self, abc_file):
        '''
        parse the raw abc file (txt) and return a list of songs with selected features (notes, beats, etc,.)
        :param abc_file: name of the file to be parsed, or 'all' if parsing all files in the directory. e.g. 'jigs.abc'
        :return: output: a list of parsed songs, each in a dictionary
        '''

        output = []
        if abc_file != 'all':
            path = 'data/' + self.dataset_name + '/' + abc_file
            abc = music21.converter.parse(path)
        else:
            pass # TODO


        for song_idx, song in enumerate(abc):
            dic = {'song_id': song_idx, 'note': None} # TODO: more features to be added
            dic['note'] = [note.nameWithOctave for note in song.flat.notes if 'nameWithOctave' in dir(note)]
            output.append(dic)

        return(output)


ds = NottinghamDataset()
outs = ds.parse_abc_txt('jigs.abc')