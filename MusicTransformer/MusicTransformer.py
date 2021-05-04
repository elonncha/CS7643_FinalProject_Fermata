# defining the transformer model
# Iman Haque CS7643 Spring 2021 final project

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# transformer model class
class MusicTransformer(nn.Module):

	# initialize the class
	def __init__(self, ntokens, ninputs, nheads, nhiddens, nlayers, dropout = 0.5):
		super(MusicTransformer, self).__init__()

		self.model_type = 'Transformer'
		self.pos_encoder = PositionalEncoding(ninputs, dropout)
		encoder_layers = TransformerEncoderLayer(ninputs, nheads, nhiddens, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.encoder = nn.Embedding(ntokens, ninputs)
		self.ninputs = ninputs
		self.decoder = nn.Linear(ninputs, ntokens)

		self.init_weights()

	# mask future tokens so sequence is learned on previous ones
	def generate_square_subsequent_mask(self, size):
		# returns diagonal true/false mask, true on lower side
		mask = (torch.triu(torch.ones((size, size))==1)).transpose(0,1)	
		# convert True/False to 1.0/0.0, then convert 0.0 to -inf and 1.0 to 0.0
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	# initialize the parameters
	def init_weights(self):
		init_range = 0.1
		self.encoder.weight.data.uniform_(-init_range, init_range)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-init_range, init_range)

	# forward pass on the transformer
	def forward(self, src, src_mask):
		# first embedding
		src = self.encoder(src)*math.sqrt(self.ninputs)
		# adding positional embedding
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, src_mask)
		output = self.decoder(output)
		return output

# adding the positional encoding as a class
class PositionalEncoding(nn.Module):

	def __init__(self, dim_model, dropout = 0.1, max_len = 256):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p = dropout)

		# init positional encoding as zero tensor
		pos_enc = torch.zeros((max_len, dim_model))
		# init position as column vec of arange of length max_len
		position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
		# vec of every 2 up to dim_model, log'd and exp'd
		div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
		
		pos_enc[:, 0::2] = torch.sin(position * div_term)
		pos_enc[:, 1::2] = torch.cos(position * div_term)
		# ^ this applies sin to every odd column and cos to every even column

		pos_enc = pos_enc.unsqueeze(0).transpose(0,1)
		self.register_buffer('pos_enc', pos_enc)

	def forward(self, x):
		# add the positional encoding and dropout
		x = x + self.pos_enc[:x.size(0), :] 
		return self.dropout(x)








