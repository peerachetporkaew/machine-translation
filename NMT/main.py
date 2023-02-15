import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

import random

from encoder import *
from decoder import *
from nmt import *
from multiheaded_attention import *
from utils import *
from generator import *
from criterion import *
from optimizer import *

import time

use_cuda = False
update_count = 0
valid_interval = 100
MAX_LEN=20

def c(x):
    return copy.deepcopy(x)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    print("Multihead...")
    attn = MultiHeadedAttention(h, d_model)

    print("Position...")
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    print("PositionEncoding...")
    position = PositionalEncoding(d_model, dropout)

    print("Build Encoder-Decoder")
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(d_model,h, d_ff, N, dropout),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    print("Init Params...")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    print(model)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, src_pad=0, trg_pad=0):
        self.src = src
        self.src_mask = (src != src_pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, trg_pad)

            if use_cuda:
                self.trg_mask = self.trg_mask.cuda()

            self.ntokens = (self.trg_y != trg_pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask



def run_epoch(data_iter, model, loss_compute, optimizer):
    global valid_interval
    global update_count
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    print("Start Epoch...")
    for i, batch in enumerate(data_iter):
        try:
            out = model.forward(batch.src, batch.trg, 
                                batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            update_count += 1
            if i % 2 == 0:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                        (i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
            
            if (i+1) % valid_interval == 0:
                #torch.save(model.state_dict(), "../checkpoints/update_%d.pt"%update_count)
                test(model)

            if use_cuda:
                torch.cuda.empty_cache()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                raise e

    print("End epoch")
    test(model)
    return total_loss / total_tokens

#new data gen
def data_gen_v2(padded_data_th,padded_data_en, src_pad, trg_pad, batch_size=64):
    global use_cuda
    #print(src_pad, trg_pad)
    size = len(padded_data_en)
    sh = [i for i in range(len(padded_data_en))]
    random.shuffle(sh)
    temp_th = [padded_data_th[t] for t in sh]
    temp_en = [padded_data_en[t] for t in sh]
    padded_data_th = temp_th
    padded_data_en = temp_en
    for i in range(0, size, batch_size):
        src = torch.LongTensor(padded_data_th[i: i+batch_size])
        tgt = torch.LongTensor(padded_data_en[i: i+batch_size])
        src.requires_grad = False
        tgt.requires_grad = False
        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
        batch = Batch(src, tgt, src_pad, trg_pad)
        yield batch

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys, 
                           subsequent_mask(ys.size(1))
                                    .type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        if next_word == end_symbol:
            break
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def create_padded_dataset(data_th, data_en, th2id, en2id, maxlen=80):

    padded_th = []
    padded_en = []

    for sentence in data_th:
        #print(sentence)
        token = sentence.strip().split()[0:maxlen-2]
        #print(len(token))
        if len(token) < maxlen-2:
            pad = ["<blank>"] * (maxlen-2 - len(token) + 1)
            token.append("<end>")
            token.extend(pad)
        else:
            
            token = ["<start>"] + token + ["<end>"]

        #print(len(token), maxlen)
        assert len(token) == maxlen

        temp = [th2id.get(t,th2id["<unk>"]) for t in token]
        padded_th.append(temp)

    for sentence in data_en:
        token = sentence.strip().split()[0:maxlen-2]
        if len(token) <= maxlen-2:
            pad = ["<blank>"] * (maxlen-2 - len(token))
            token.append("<end>")
            token.extend(pad)
            token = ["<start>"] + token
        else:
            token = ["<start>"] + token + ["<end>"]

        temp = [en2id.get(t,en2id["<unk>"]) for t in token]
        padded_en.append(temp)
    
    return padded_th, padded_en


def build_dictionary():
    th2id = {}
    id2th = {}

    with open("./dataset/train.ja.vocab.4k", mode='r') as file:
        data_th = file.readlines()

        th2id["<blank>"] = len(th2id)
        id2th[th2id["<blank>"]] = "<blank>"

        th2id["<start>"] = len(th2id)
        id2th[th2id["<start>"]] = "<start>"
        
        th2id["<end>"] = len(th2id)
        id2th[th2id["<end>"]] = "<end>"

        th2id["<unk>"] = len(th2id)
        id2th[th2id["<unk>"]] = "<unk>"

        for id , item in enumerate(data_th):
            w = item.strip().split(" ")[0]
            th2id[w] = len(th2id)
            id2th[th2id[w]] = w

    en2id = {}
    id2en = {}

    with open("./dataset/train.en.vocab.4k", mode='r') as file:
        data_en = file.readlines()

        en2id["<start>"] = len(en2id)
        id2en[en2id["<start>"]] = "<start>"
        
        en2id["<end>"] = len(en2id)
        id2en[en2id["<end>"]] = "<end>"

        en2id["<blank>"] = len(en2id)
        id2en[en2id["<blank>"]] = "<blank>"

        en2id["<unk>"] = len(en2id)
        id2en[en2id["<unk>"]] = "<unk>"

        for id , item in enumerate(data_en):
            w = item.strip().split(" ")[0]
            en2id[w] = len(en2id)
            id2en[en2id[w]] = w

    return th2id,id2th,en2id,id2en

def load_dataset_and_train():
    global use_cuda
    max_len = MAX_LEN
    sample_size = 50000

    with open("./dataset/train.en", mode='r') as file:
        data_en = file.readlines()[:sample_size]

    with open("./dataset/train.ja", mode='r') as file:
        data_th = file.readlines()[:sample_size]

    corpus_en =[]
    corpus_th =[]

    for en,th in zip(data_en,data_th):
        len_en = len(en.strip().split())
        len_th = len(th.strip().split())
        if len_en < max_len and len_th < max_len:
            corpus_en.append(en)
            corpus_th.append(th)

    print("Corpus size : ", len(corpus_en))
    th2id,id2th,en2id,id2en = build_dictionary()
    max_len = 20
    batch_size = 20

    padded_data_th, padded_data_en = create_padded_dataset(corpus_th, corpus_en, th2id, en2id, max_len)

    print("Setting up model ....")

    criterion = LabelSmoothing(size=len(en2id), padding_idx=en2id["<blank>"], smoothing=0.1)
    
    print("Make Model...")
    model = make_model(len(th2id), len(en2id), N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
    
    #model.load_state_dict(torch.load("./checkpoints/update_150834.pt"))
    print("Build Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007, betas=(0.9, 0.98), eps=1e-9)

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 4000, optimizer)

    print("Begin Training...")

    for epoch in range(200):
        print("Entering epoch : %d" % epoch)
        model.train()
        
        run_epoch(data_gen_v2(padded_data_th,padded_data_en,th2id["<blank>"], en2id["<blank>"], batch_size), model, SimpleLossCompute(model.generator, criterion, model_opt), optimizer)
        
        torch.save(model.state_dict(), "../checkpoints/chk_%d.pt"%epoch)

def encode_line(input, th2id):    
    word = [th2id["<start>"]] + [th2id.get(t,th2id["<unk>"]) for t in input.split()] + [th2id["<end>"]]

    #fill blank
    for i in range(len(word),MAX_LEN):
        word.append(th2id["<blank>"])

    src = torch.LongTensor(word)
    src_mask = (src != th2id["<blank>"]).unsqueeze(-2)
    return src, src_mask

def test_one(dictionaries, model, input, use_cuda ):
    th2id,id2th,en2id,id2en = dictionaries
    model.eval() 
    print("INPUT : ", input.strip()) #input = "Thank you for comming by ."
    src, src_mask = encode_line(input.strip(), th2id)
    src = src.unsqueeze(0)
    #print(src)
    src_mask = src_mask.unsqueeze(0)
    print(src)
    print(src_mask)
    if use_cuda:
        print("CUDA")
        model = model.cuda()
        src = src.cuda()
        src_mask = src_mask.cuda()

    out = greedy_decode(model,src,src_mask,MAX_LEN,en2id["<start>"], en2id["<end>"])
    output = out[0].tolist()
    trg = [id2en[id] for id in output[1:]]
    out = " ".join(trg).replace(" <blank>","")
    print("OUTPUT : ",out)
    print("---")
    return out.strip()
    
def test(model):
    global use_cuda

    model.eval()
    dictionaries = build_dictionary()
    with open("./dataset/test.ja", mode='r') as file:
        data_th = file.readlines()
    random.shuffle(data_th)
    for i in range(0,10):
        test_one(dictionaries, model, data_th[i], use_cuda)
    model.train()

def gen_output():

    global use_cuda
    th2id,id2th,en2id,id2en = build_dictionary()
    model1 = make_model(len(th2id), len(en2id), N=6, d_model=512, d_ff=2048, h=8)   
    model1.load_state_dict(torch.load("../checkpoints/update_166938.pt",map_location="cpu"))

    if use_cuda:
        model = model1.cuda()

    model.eval()
    dictionaries = build_dictionary()
    with open("../dataset/test3k.bpe.th", mode='r') as file:
        data_th = file.readlines()
    
    fout = open("output.txt","w")

    for i in range(0,len(data_th)):
        print(i)
        out = test_one(dictionaries, model, data_th[i], use_cuda)
        fout.writelines(out + "\n")

    fout.close()

    os.system("cat output.txt | sacrebleu ../dataset/test3k.bpe.en")
    return None
    

def eval():
    global use_cuda
    th2id,id2th,en2id,id2en = build_dictionary()
    model = make_model(len(th2id), len(en2id), N=6, d_model=512, d_ff=2048, h=8)
    model.load_state_dict(torch.load("../checkpoints/update_9402.pt"))
    src, src_mask = encode_line("ขอบคุณ ที่ มา หา พวกเรา", th2id)
    src = src.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)
    print(src)
    print(src_mask)

    if use_cuda:
        model = model.cuda()
        src = src.cuda()
        src_mask = src_mask.cuda()

    out = greedy_decode(model,src,src_mask,MAX_LEN,en2id["<start>"])
    output = out[0].tolist()

    trg = [id2en[id] for id in output]
    print(" ".join(trg).replace(" <blank>",""))


if __name__ == "__main__":
    print(torch.__version__) # 1.7+
    #gen_output()
    load_dataset_and_train()
    #eval()
