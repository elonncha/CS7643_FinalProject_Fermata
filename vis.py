from data_util import *
import matplotlib.pyplot as plt

full_data = parse_folk_by_txt(meter = '4/4', seq_len_min = 0, seq_len_max = 9999)

length = [len(x) for x in full_data[0]]

plt.figure()
plt.hist(length, bins = 100)
plt.xlabel('sequence length')
plt.title('Histogram of seuqnce length, folkrnn dataset')
plt.savefig('seqlenhist.png')
plt.show()