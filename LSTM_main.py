from data_util import *
import torch
import torch.nn as nn
import pickle
import time
from torch.utils.data import DataLoader
from Seq2Seq_LSTM.Encoder import Encoder
from Seq2Seq_LSTM.Decoder import Decoder
from Seq2Seq_LSTM.Seq2Seq import Seq2Seq

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch, data_loader, model, optimizer, criterion):
    
    iter_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()

    for i, data in enumerate(data_loader):

        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            device = torch.device('cuda')
        else:
            device = 'cpu'

        note_past, measure_past, note_future, measure_future, note_target = data
        print('from data loader: ', note_past.shape, measure_past.shape, note_future.shape, measure_future.shape, note_target.shape)
        outputs = model.forward(note_past, measure_past, note_future, measure_future, note_target)
        print(outputs.shape)

        prediction = torch.argmax(outputs, dim=2)
        print(prediction.shape)

        predicted_note = np.empty_like(prediction, dtype = 'str')
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                predicted_note[i,j] = note_dic[prediction[i,j].item()]
        print(predicted_note.shape)


        loss = criterion(output, target)

        optimizer.zero_grad()
        print('Loss: ', loss)

    #     loss.backward()
    #     optimizer.step()

    #     losses.update(loss.detach().item(), output.shape[0])

    #     iter_time.update(time.time() - start)
    #     if idx % 10 == 0:
    #         print(('Epoch: [{0}][{1}/{2}]\t'
    #                'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
    #                'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
    #                .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses))
    # avg_loss = losses.avg.item()
    # perplexity = np.exp(avg_val_loss)

    return losses.val.item(), avg_loss, perplexity


# data load
# note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()
# data_train, data_test, data_val = train_test_val_split(note_past, note_target, note_future, measure_past, measure_mask, measure_future, song_id)
# data = [data_train, data_val, data_test]
# ds_names = ['train', 'val', 'test']

# # dump to pickle
# for i, ds_name in enumerate(ds_names):
#     path = './dataset_split/'+ds_name
#     with open(path, 'wb') as pickle_w:
#         write = {b'note_past': data[i][0],
#                  b'note_future': data[i][2],
#                  b'measure_past': data[i][3],
#                  b'measure_future': data[i][5],
#                  b'target': data[i][1]}
#         pickle.dump(write, pickle_w)
#     # open test
#     with open(path, 'rb') as pickle_r:
#         dict = pickle.load(pickle_r, encoding='bytes')
#         print(ds_name, dict[b'target'])





# Temp settings before implementing Ray Tune for hyperparameter tuning
epochs = 1 # 100
# Inpaint paper notes:
    # The MeasureVAE model was pre-trained using single measures following the standard VAE optimization equa- tion [26] with the β-weighting scheme [41,42]. 
    # The Adam algorithm [43] was used for model training, with a learning rate of 1e−3, β1 = 0.9, β2 = 0.999, and ε = 1e−8.
hp = {'batch_size': 1,#128,
      'lr': 0.003,
      'reg': 0.001,
      'emb_size': 10,
      'enc_hidden_size': 256,
      'dec_hidden_size': 256,
      'dropout': 0.2
     }

data_root = './dataset_split'
results_root = './results/biLSTM'


torch.manual_seed(0)
batch_size = hp['batch_size']
train_data = INPAINT(data_root, ds_type='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
note_vocab_size = train_data.note_vocab_size()
measure_vocab_size = train_data.measure_vocab_size()
target_note_vocab_size = train_data.target_note_vocab_size()
seq_length_past, seq_length_future = train_data.seq_length()
output_size = note_vocab_size

val_data = INPAINT(data_root, ds_type='val')
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, drop_last=True) # change batch size ?

print(note_vocab_size, measure_vocab_size, seq_length_past, seq_length_future)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# temp for testing
past_encoder = Encoder(vocab_size=note_vocab_size, max_measure=measure_vocab_size, seq_len=seq_length_past,
                       emb_size=hp['emb_size'],
                       encoder_hidden_size=hp['enc_hidden_size'], decoder_hidden_size=hp['dec_hidden_size'],
                       dropout=hp['dropout'])

future_encoder = Encoder(vocab_size=note_vocab_size, max_measure=measure_vocab_size, seq_len=seq_length_future,
                         emb_size=hp['emb_size'],
                         encoder_hidden_size=hp['enc_hidden_size'], decoder_hidden_size=hp['dec_hidden_size'],
                         dropout=hp['dropout'])

decoder = Decoder(emb_size=hp['emb_size'], decoder_hidden_size=past_encoder.decoder_hidden_size*4, output_size=target_note_vocab_size, dropout=hp['dropout'])

model = Seq2Seq(past_encoder, future_encoder, decoder, device = 'cpu')

if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), hp['lr'], weight_decay=hp['reg'])

for epoch in range(epochs):

    train_loss, avg_train_loss, train_perplexity = train(epoch, train_loader, model, optimizer, criterion)

    # val_loss, avg_val_loss, val_perplexity = evaluate(eval_loader, batch_size, encoder, decoder, model, 'val', criterion, epoch=epoch)

    # print('Train:\n', train_loss, avg_train_loss, train_perplexity)
    # print('Val:\n', val_loss, avg_val_loss, val_perplexity)



