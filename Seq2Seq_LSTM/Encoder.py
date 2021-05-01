import torch
import torch.nn as nn

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
    """
    def __init__(self, vocab_size, max_measure, seq_len,
                 emb_size,
                 encoder_hidden_size, decoder_hidden_size, dropout = 0.2):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.max_measure = max_measure
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        ''' embedding '''
        self.note_emb = nn.Embedding(num_embeddings=self.vocab_size,
                                     embedding_dim = self.emb_size)
        self.measure_emb = nn.Embedding(num_embeddings=self.max_measure+3,
                                        embedding_dim = self.emb_size,
                                        padding_idx = 0)
        self.position_emb = nn.Embedding(num_embeddings=self.seq_len,
                                         embedding_dim=self.emb_size)

        ''' LSTM '''
        self.rec = nn.LSTM(input_size = self.emb_size,
                           hidden_size=self.encoder_hidden_size,
                           batch_first=True,
                           bidirectional=True)

        ''' FC '''
        self.fc1 = nn.Linear(in_features=self.encoder_hidden_size,
                             out_features=self.encoder_hidden_size)
        self.fc2 = nn.Linear(in_features=self.encoder_hidden_size,
                             out_features=self.decoder_hidden_size)

        ''' Drop '''
        self.drop = nn.Dropout(p = dropout)


    def embedding(self, note, measure):

        note_embedding = self.note_emb(note)

        measure_embedding = self.measure_emb(measure+3)

        position = torch.repeat_interleave(torch.arange(0, self.seq_len).unsqueeze(dim=0), measure.shape[0], dim=0)
        position_embedding = self.position_emb(position)

        return(note_embedding + measure_embedding + position_embedding)


    def forward(self, note, measure):

        # embedding and dropout
        emb = self.embedding(note, measure)
        emb = self.drop(emb)

        # recurrent, output: [batch_size x seq_len x encoder_hidden_size]
        LSTM = self.rec(emb)
        output = LSTM[0]
        hidden = LSTM[1][0]

        # linear -relu -linear
        hidden = self.fc2(torch.relu(self.fc1(hidden)))

        # tanh, hidden: [2 x batch_size x decoder_hidden_size]
        hidden = torch.tanh(hidden)

        # Bidirectional, concatenate bidirectional hidden states into one with shape [batch_size, decoder_hidden_size * 2]
        hidden = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim = 1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden

