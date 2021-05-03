import torch
import torch.nn as nn

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, output_size, emb_size, decoder_hidden_size, dropout = 0.2):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size

        self.emb = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.emb_size)

        ''' 2) '''
        self.rec = nn.LSTM(input_size=self.emb_size, hidden_size=self.decoder_hidden_size, batch_first=True)

        ''' 3) '''
        self.fc1 = nn.Linear(in_features=self.decoder_hidden_size, out_features=self.output_size)

        ''' 4) '''
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        embedding = self.emb(input)
        embedding = self.drop(embedding)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LSTM = self.rec(embedding, (hidden, torch.zeros_like(hidden).to(device)))
        output = LSTM[0]
        hidden = LSTM[1][0]

        output = output[:,0,:]
        output = torch.log_softmax(self.fc1(output), dim = 1)

        return output, hidden
