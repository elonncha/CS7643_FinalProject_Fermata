from data_util import *



# load original data
note, measure, song_id = parse_folk_by_txt(meter = '4/4', seq_len_min = 256, seq_len_max = 256+32)


# slice by fractions
note_past, note_target, note_future = slicing_by_fraction(note, past_fraction = 0.3, future_fraction = 0.3)
measure_past, measure_mask, measure_future = slicing_by_fraction(measure, past_fraction = 0.3, future_fraction = 0.3)


# add paddings
note_past, note_future = add_padding(note_past, position = 'left'), add_padding(note_future, position = 'right')
measure_past, measure_future = add_padding(measure_past, position = 'left'), add_padding(measure_future, position = 'right')


