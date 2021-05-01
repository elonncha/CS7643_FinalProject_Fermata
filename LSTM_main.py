from data_util import *
import torch
import pickle
from torch.utils.data import DataLoader

# data load
note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()
data_train, data_test, data_val = train_test_val_split(note_past, note_target, note_future, measure_past, measure_mask, measure_future, song_id)
data = [data_train, data_val, data_test]
ds_names = ['train', 'val', 'test']

# dump to pickle
for i, ds_name in enumerate(ds_names):
    path = './dataset_split/'+ds_name
    with open(path, 'wb') as pickle_w:
        write = {b'note_past': data[i][0],
                 b'note_future': data[i][2],
                 b'measure_past': data[i][3],
                 b'measure_future': data[i][5],
                 b'target': data[i][1]}
        pickle.dump(write, pickle_w)
    # open test
    with open(path, 'rb') as pickle_r:
        dict = pickle.load(pickle_r, encoding='bytes')
        print(ds_name, dict[b'target'])





# Temp settings before implementing Ray Tune for hyperparameter tuning
epochs = 3 # 100
# Inpaint paper notes:
    # The MeasureVAE model was pre-trained using single measures following the standard VAE optimization equa- tion [26] with the β-weighting scheme [41,42]. 
    # The Adam algorithm [43] was used for model training, with a learning rate of 1e−3, β1 = 0.9, β2 = 0.999, and ε = 1e−8.
hp = {'batch_size': 10,#128,
      'lr': 0.003,
      'reg': 0.001,
    #   'emb_size': 10,
      'hidden_size': 256,
      'dropout': 0.2
     }

data_root = './dataset_split'
results_root = './results/biLSTM'
model = 'biLSTM'


torch.manual_seed(0)
batch_size = hp['batch_size']
train_data = INPAINT(data_root, ds_type='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
train_target = train_data.target
input_size = train_data.vocab_size()

seq_length_past, seq_length_future = train_data.seq_length()
output_size = input_size

val_data = INPAINT(data_root, ds_type='val')
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, drop_last=True) # change batch size ?
val_target = val_data.target

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if model == 'biLSTM':
#     model = biLSTM(batch_size, input_size, seq_length_past, seq_length_future, hp['hidden_size'], hp['dropout'])
# if torch.cuda.is_available():
#     model = model.cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), hp['lr'], weight_decay=hp['reg'])

# for epoch in epochs:
#
