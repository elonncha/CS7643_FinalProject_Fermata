from data_util import *
from Seq2Seq_LSTM.Encoder import Encoder
from Seq2Seq_LSTM.Decoder import Decoder
from Seq2Seq_LSTM.Seq2Seq import Seq2Seq
import torch

#data load
note_past, note_target, note_future, measure_past, measure_mask, measure_future, note_dic, song_id = load_data()
data_train, data_test, data_val = train_test_val_split(note_past, note_target, note_future, measure_past, measure_mask, measure_future, song_id)
data = [data_train, data_val, data_test]



past_note = torch.tensor(data_train[0])[0:128,:] # 128*87
past_measure = torch.tensor(data_train[3])[0:128,:] # 128*87
future_note = torch.tensor(data_train[2])[0:128,:] # 128*88
future_measure = torch.tensor(data_train[5])[0:128,:] # 128*88
target = torch.tensor(data_train[1])[0:128,:] # 128*117




past_encoder = Encoder(vocab_size = note_dic.shape[0], max_measure = 43, seq_len = past_note.shape[1],
                       emb_size = 10,
                       encoder_hidden_size = 256,decoder_hidden_size = 256,
                       dropout = 0.2)

future_encoder = Encoder(vocab_size = note_dic.shape[0], max_measure = 43, seq_len = future_note.shape[1],
                         emb_size = 10,
                         encoder_hidden_size = 256,decoder_hidden_size = 256,
                         dropout = 0.2)

decoder = Decoder(emb_size = 10, decoder_hidden_size = past_encoder.decoder_hidden_size * 4, output_size = 93, dropout = 0.2)



s = Seq2Seq(past_encoder, future_encoder,decoder, device = 'cpu')
outputs = s.forward(past_note, past_measure, future_note, future_measure, target)


prediction = torch.argmax(outputs, dim = 2)

predicted_note = np.empty_like(prediction, dtype = 'str')
for i in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
        predicted_note[i,j] = note_dic[prediction[i,j].item()]


