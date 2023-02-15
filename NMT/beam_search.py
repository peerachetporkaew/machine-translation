import torch
from utils import Batch,subsequent_mask
from icecream import ic
import time,gc

def beam_decode(model,beam_size, src, src_mask, max_len, start_symbol, end_symbol, pad_symbol):
    #print(beamsize)
    model.eval()
    bsize, seql = src.size()[:2]
    beam_size2 = beam_size*beam_size
    bsizeb2 = bsize * beam_size2
    real_bsize = bsize * beam_size

    #Run forward pass
    """
        1. Encode input 
        2. Prepare target <start> symbol
        3. Feed to model
    """
    memory = model.encode(src, src_mask) # 1

    ys = torch.ones(bsize, 1).fill_(start_symbol).type_as(src.data) # 2

    #ic(src_mask)
    #ic(ys)

    out = model.decode(memory, src_mask,ys,
                       subsequent_mask(ys.size(1)).type_as(src.data)) #3
    out = model.generator(out[:, -1]) #3

    scores, wds = out.topk(beam_size, dim=-1)
    scores = scores.squeeze(1)
    sum_scores = scores
    wds = wds.view(real_bsize, 1)

    _ys = torch.ones(real_bsize,1).fill_(start_symbol).type_as(src.data)

    trans = torch.cat((_ys,wds),1)

    # done_trans: (bsize, beam_size)

    done_trans = wds.view(bsize, beam_size).eq(end_symbol)

    _src_pad_mask = None if src_mask is None else src_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

    #memory ()
    memory = memory.repeat(1, beam_size, 1).view(real_bsize, seql, -1)

    fill_pad = True

    for step in range(1, max_len):
        out = model.decode(memory, _src_pad_mask,trans,
                       subsequent_mask(trans.size(1)).type_as(src.data)) #3
        out = model.generator(out[:, -1]) #3
        
        out = out.view(bsize, beam_size, -1)

        #Find top k^2 candidates and calcuate scores
        _scores, _wds = out.topk(beam_size, dim=-1)
        _scores = (_scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size))

        scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
        _tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
        sum_scores = scores

        wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)
        _inds = (_inds // beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

        trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_symbol) if fill_pad else wds), 1)

        done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(2).squeeze(1)).view(bsize, beam_size)

        if done_trans.all().item():
            return trans.view(bsize, beam_size, -1).select(1, 0).tolist()
    
    
    

    out = trans.view(bsize, beam_size, -1).select(1, 0).tolist()
    del trans, done_trans, memory, scores, _scores
    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    return out 