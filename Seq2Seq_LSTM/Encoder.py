import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
    """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type

        ''' 1) '''
        self.emb = nn.Embedding(num_embeddings=self.input_size, embedding_dim = self.emb_size)

        ''' 2) '''
        self.rec = nn.LSTM(input_size = self.emb_size, hidden_size= self.encoder_hidden_size, batch_first=True)

        ''' 3) '''
        self.fc1 = nn.Linear(in_features = self.encoder_hidden_size, out_features=self.encoder_hidden_size)
        self.fc2 = nn.Linear(in_features = self.encoder_hidden_size, out_features=self.decoder_hidden_size)

        ''' 4) '''
        self.drop = nn.Dropout(p = dropout)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len, input_size)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        
        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################

        # embedding and dropout
        embedding = self.emb(input)
        embedding = self.drop(embedding)

        # recurrent, output: [batch_size x seq_len x encoder_hidden_size]
        if self.model_type == 'RNN':
            output, hidden = self.rec(embedding) # embedding: [batch x seq_len x embedding_size]

        if self.model_type == 'LSTM':
            LSTM = self.rec(embedding)
            output = LSTM[0]
            hidden = LSTM[1][0]


        # linear -relu -linear
        hidden = self.fc2(torch.relu(self.fc1(hidden)))

        # return, hidden [1 x batch x encoder_hidden_size]
        hidden = torch.tanh(hidden)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden

