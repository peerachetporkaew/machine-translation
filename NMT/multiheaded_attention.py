import torch
import math
import torch.nn as nn
from utils import clones
import torch.nn.functional as F
from utils import with_incremental_state
from typing import Dict, Optional, Tuple
from torch import Tensor
import time

from icecream import ic

#ic.disable()

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #ic(scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    #ic(p_attn.size())
    if dropout is not None:
        p_attn = dropout(p_attn)
    #ic(p_attn.size())
    #ic(value.size())
    out = torch.matmul(p_attn, value)
    #ic(out.size())
    return out, p_attn

@with_incremental_state
class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(0,4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, is_decoder_self_attention=False, 
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None):

        """
            query : (nbatches, seqlen, dim)
            key   : (nbatches, seqlen, dim)
            value : (nbatches, seqlen, dim)

        """

        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        if incremental_state is None or is_decoder_self_attention == False:
            # 1) Do all the linear projections in batch from d_model => h x d_k 
            # Incremental state shoule apply linear only for last time step
            query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]
            
            # 2) Apply attention on all the projected vectors in batch. 
            # If Incremental state is not None, concat the saved_state (prev_key, prev_value) before doing attention.


            #TODO：Insert state here

            x, self.attn = attention(query, key, value, mask=mask, 
                                    dropout=self.dropout)

            #TODO：Save new key, value to the incremental state

            
            # 3) "Concat" using a view and apply a final linear. 
            x = x.transpose(1, 2).contiguous() \
                .view(nbatches, -1, self.h * self.d_k) #nbatch, seql, hidden_dim 
            return self.linears[-1](x)
        else:
            # 1) Do all the linear projections in batch from d_model => h x d_k 
            # Incremental state shoule apply linear only for last time step
            assert is_decoder_self_attention == True

            saved_state = self._get_input_buffer(incremental_state)
            prev_key = None
            prev_value = None 
            if saved_state is not None:
                #saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
                if "prev_key" in saved_state:
                    ic(saved_state["prev_key"].size())
                    prev_key = saved_state["prev_key"]

                if "prev_value" in saved_state:
                    ic(saved_state["prev_value"].size())
                    prev_value = saved_state["prev_value"]

            temp_query = query 
            temp_key   = key
            temp_value = value 

            if prev_key is not None and prev_value is not None:
                query_1 = temp_query[:,-1,:]
                key_1   = temp_key[:,-1,:]
                value_1 = temp_value[:,-1,:]

                query_1, key_1, value_1 = \
                    [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                    for l, x in zip(self.linears, (query_1, key_1, value_1))]
            else:
                query, key, value = \
                    [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                    for l, x in zip(self.linears, (query, key, value))]
            
            
            # 2) Apply attention on all the projected vectors in batch. 
            
            if prev_key is not None:
                cat_key = torch.cat([prev_key,key_1],dim=2)
                ic(cat_key.size())
                cat_value = torch.cat([prev_value,value_1],dim=2)
                saved_state["prev_key"]    = cat_key    # (nbatch,num_head,seql,dim)
                saved_state["prev_value"]  = cat_value  # (nbatch,num_head,seql,dim)
                incremental_state = self._set_input_buffer(incremental_state, saved_state)



                #ic(cat_value.size())
                #ic(query_1.size())
                x_1, self.attn = attention(query_1, cat_key, cat_value, mask=None, dropout=self.dropout) #No masking
                #ic(x_1.size())
                #TODO：Save new key, value to the incremental state
            else:

                #TODO：Insert state here
                saved_state["prev_key"]    = key    # (nbatch,num_head,seql,dim)
                saved_state["prev_value"] = value  # (nbatch,num_head,seql,dim)
                incremental_state = self._set_input_buffer(incremental_state, saved_state)

                #ic("original query", query.size())
                #ic("original key", key.size())
                #ic("original value", value.size())

                x, self.attn = attention(query, key, value, mask=mask, 
                                        dropout=self.dropout)

                #ic("original x", x.size())
                

            # 3) "Concat" using a view and apply a final linear. 
            if prev_key is not None:
                x_1 = x_1.transpose(1, 2).contiguous() \
                .view(nbatches, -1, self.h * self.d_k) #nbatch, seql, hidden_dim 
                #ic(x_1.size())
                #ic((x[:,-1,:] - x_1).sum())
                #time.sleep(1)

                return self.linears[-1](x_1)
            
            else:
                x = x.transpose(1, 2).contiguous() \
                .view(nbatches, -1, self.h * self.d_k) #nbatch, seql, hidden_dim 

                #ic(x.size())
                return self.linears[-1](x)

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
