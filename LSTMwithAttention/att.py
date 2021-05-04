import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from helpers import d

class AttentionError(Exception):
    pass

class MultiheadedAttention(nn.Module):

    def __init__(self, d_model, heads=8, dropout=0.1, relative_pos=True):

        super().__init__()
        if d_model % heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = 1024
        #relative positional embeddings
        self.relative_pos = relative_pos
        if relative_pos:
            self.Er = torch.randn([heads, self.max_length, s],
                    device=d())
        else:
            self.Er = None

    def forward(self, x, mask):
  
        b, t, e = x.size()
        h = self.heads
     
        s = e // h
        #start index of position embedding
        embedding_start = self.max_length - t
        x = x.view(b,t,h,s)
        queries, keys, values = [w(x).transpose(1,2)
                for w, x in zip(self.linears, (x,x,x))]
        if self.relative_pos:
            #apply same position embeddings across the batch
            #Is it possible to apply positional self-attention over
            #only half of all relative distances?
            Er  = self.Er[:, embedding_start:, :].unsqueeze(0)
            QEr = torch.matmul(queries, Er.transpose(-1,-2))
            QEr = self._mask_positions(QEr)
            #Get relative position attention scores
            #combine batch with head dimension
            SRel = self._skew(QEr).contiguous().view(b*h, t, t)
        else:
            SRel = torch.zeros([b*h, t, t], device=d())
        queries, keys, values = map(lambda x: x.contiguous()\
                .view(b*h, t, s), (queries, keys, values))
        #Compute scaled dot-product self-attention
        #scale pre-matrix multiplication   
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))

        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores + SRel
        #(b*h, t, t)

        subsequent_mask = torch.triu(torch.ones(1, t, t, device=d()),
                1)
        scores = scores.masked_fill(subsequent_mask == 1, -1e9)
        if mask is not None:
            mask = mask.repeat_interleave(h, 0)
            wtf = (mask == 0).nonzero().transpose(0,1)
            scores[wtf[0], wtf[1], :] = -1e9

        
        attn_probs = F.softmax(scores, dim=2)
        attn_probs = self.dropout(attn_probs)
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        return self.recombine_heads(out)


    def _mask_positions(self, qe):
       
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=d()), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]
