import torch
import torch.nn as nn

class biLSTMEncoder(nn.Module):
    def __init__(self, batch_size, input_size, seq_length_past, seq_length_future, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(biLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.seq_length_past = seq_length_past
        self.seq_length_future = seq_length_future
        # self.emb_size = emb_size
        self.enc_hidden_size = enc_hidden_size

        self.note_embed_past = nn.Embedding(num_embeddings=input_size, embedding_dim=enc_hidden_size)
        self.posit_embed_past = nn.Embedding(num_embeddings=seq_length_past, embedding_dim=enc_hidden_size)
        self.note_embed_future = nn.Embedding(num_embeddings=input_size, embedding_dim=enc_hidden_size)
        self.posit_embed_future = nn.Embedding(num_embeddings=seq_length_future, embedding_dim=enc_hidden_size)

        self.lstm_forward = nn.LSTMCell(enc_hidden_size, enc_hidden_size)
        self.lstm_backward = nn.LSTMCell(enc_hidden_size, enc_hidden_size)
        # self.linear = nn.Linear(enc_hidden_size, dec_hidden_size)
        # self.lstm = nn.LSTMCell(enc_hidden_size * 2, enc_hidden_size * 2) # for decoder
        self.dropout = nn.Dropout(dropout)

        self.hs_forward, self.cs_forward, self.hs_backward, self.cs_backward = [nn.Parameter(torch.zeros(batch_size, self.enc_hidden_size)) for i in range(4)]
        # self.hs_lstm, self.cs_lstm = [nn.Parameter(torch.zeros(batch_size, self.enc_hidden_size * 2)) for i in range(2)]

        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x):

        hidden_forward, hidden_backward = [], []

        hs_forward, cs_forward, hs_backward, cs_backward = self.hs_forward, self.cs_forward, \
                                                           self.hs_backward, self.cs_backward

        past, future = x[0], x[1]

        out_past = torch.add(self.note_embed_past(past[0]), self.posit_embed_past(torch.arange(past[1].shape[1]))).permute(1, 0, 2)
        out_future = torch.add(self.note_embed_future(future[0]), self.posit_embed_future(torch.arange(future[1].shape[1]))).permute(1, 0, 2)

        for i in range(self.seq_length_past):
            hs_forward, cs_forward = self.lstm_forward(out_past[i], (hs_forward, cs_forward))
            hs_forward, cs_forward = [self.dropout(i) for i in [hs_forward, cs_forward]]
            hidden_forward.append(hs_forward)

        for i in reversed(range(self.seq_length_future)):
            hs_backward, cs_backward = self.lstm_backward(out_future[i], (hs_backward, cs_backward))
            hs_backward, cs_backward = [self.dropout(i) for i in [hs_backward, cs_backward]]
            hidden_backward.append(hs_backward)

        return hidden_forward, hidden_backward
