import torch
import torch.nn as nn
from utils import LayerNorm, SublayerConnection, clones, with_incremental_state, PositionwiseFeedForward
from multiheaded_attention import *
from typing import Dict, Optional, Tuple
from torch import Tensor

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size # d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for i in range(0,3)])

 
    def forward(self, x, memory, src_mask, tgt_mask, 
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, is_decoder_self_attention=True, incremental_state=incremental_state)) # self attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, incremental_state=None))  # cross attention
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, d_model, h , d_ff, N, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, MultiHeadedAttention(h, d_model), 
                                                  MultiHeadedAttention(h, d_model),
                                                  PositionwiseFeedForward(d_model, d_ff, dropout), dropout) for n in range(0,N)])

        #Checking uuid for incremental state
        #for i in range(0,N):
        #    print(self.layers[i].self_attn._incremental_state_id)

        self.norm = LayerNorm(self.layers[0].size)
        
    def forward(self, x, memory, src_mask, tgt_mask, incremental_state=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, incremental_state=incremental_state)
        return self.norm(x)