import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, past_encoder, future_encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.past_encoder = past_encoder.to(self.device)
        self.future_encoder = future_encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        # print('Model Device: ', device)
        # on_cuda = [next(x.parameters()).is_cuda for x in [self.past_encoder, self.future_encoder, self.decoder]]
        # print(on_cuda)

    def forward(self, note_past, measure_past, note_future, measure_future, note_target):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
        """
        outputs = None
        batch_size = note_past.shape[0]
        output_len = note_target.shape[1]

        # last hidden representation from encoder
        _, h_past = self.past_encoder(note_past, measure_past)
        _, h_future = self.future_encoder(note_future, measure_future)

        hidden = torch.cat((h_past, h_future), dim=1).unsqueeze(dim=0).to(self.device)

        # first input for the decoder
        input = note_target[:,0].unsqueeze(dim = 1).clone().detach().to(self.device)

        outputs = torch.zeros((batch_size, output_len, self.decoder.output_size)).to(self.device)

        for seq in range(output_len-1):

            # compute output
            output, hidden = self.decoder.forward(input, hidden)
            # append output to outputs [batch x output_size] (single step)
            outputs[:,seq+1,:] = output

            # manipulate output to next input
            input = torch.argmax(output, dim = 1).unsqueeze(dim = 1)

        outputs[:, 0, 16] = 1  # first input will always be <s> token
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs



        

