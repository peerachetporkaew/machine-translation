#git clone https://github.com/odashi/small_parallel_enja

import codecs 
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel
from nltk.translate.ibm_model import Counts
from nltk.translate import IBMModel1, IBMModel3
from nltk.translate import grow_diag_final_and
from nltk.translate.phrase_based import phrase_extraction
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
from math import log

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Lidstone

from nltk.translate import PhraseTable, StackDecoder

MAX = 50000
TEST_MAX = 10

def print_alignment(aln, invert=False, quiet=False):
    output = ""
    for p in sorted(aln):
        if p[0] is not None and p[1] is not None:
            if invert:
                output += str(p[1]) + "-" + str(p[0]) + " "
            else:
                output += str(p[0]) + "-" + str(p[1]) + " "
    if not quiet:
        print(output)
    return output


def load_corpus(path="../corpus/small_parallel_enja/"):
    ja = codecs.open(path + "train.ja",encoding="utf-8").read().splitlines()
    en = codecs.open(path + "train.en",encoding="utf-8").read().splitlines()

    return ja,en

def load_test_corpus(path="../corpus/small_parallel_enja/"):
    ja = codecs.open(path + "test.ja",encoding="utf-8").read().splitlines()[0:MAX]
    en = codecs.open(path + "test.en",encoding="utf-8").read().splitlines()[0:MAX]

    return ja,en

def train_word_alignment(srcL,trgL):
    
    bitext = []
    for src, trg in zip(srcL,trgL):
        bitext.append(AlignedSent(src.split(), trg.split()))

    ibm1 = IBMModel1(bitext, 5)
    ibm1.align_all(bitext)
    
    for pair in bitext:
        print_alignment(pair.alignment)
    
    return bitext

def extract_phrase(src2trg,trg2src,fout):
    all_phrase = []
    for k in range(len(src2trg)):
        print(k)
        alnF = print_alignment(src2trg[k].alignment,quiet=True)
        alnB = print_alignment(trg2src[k].alignment,invert=True,quiet=True)
        out = grow_diag_final_and(len(src2trg[k].words),
                                len(src2trg[k].mots),
                                alnF, alnB)
        #print(out)
        phrases = phrase_extraction(" ".join(src2trg[k].words)," ".join(src2trg[k].mots),out)
        for phrase in phrases:
            #print(phrase)
            all_phrase.append(phrase)

    fp = codecs.open(fout,"w",encoding="utf-8")
    for p in all_phrase:
        fp.writelines(p[2] + " ||| " + p[3] + "\n")
    fp.close()
    
def extract_phrase_from_corpus():
    # SRC = ja
    # TRG = en

    print("Loading corpus...")
    ja, en = load_corpus()

    print("Train word alignment (src->trg)")
    src2trg = train_word_alignment(ja,en)

    print("Train word alignment (trg->src)")
    trg2src = train_word_alignment(en,ja)

    extract_phrase(src2trg,trg2src,"phrase_out.txt")

def get_translation_probability(fout):
    phrase_frequency = {}
    phrase_prob = []
    fp = codecs.open("phrase_out.txt","r").readlines()

    #Count Frequency
    for line in fp:
        src, trg = line.strip().split(" ||| ")
        if src in phrase_frequency:
            if trg in phrase_frequency[src]:
                phrase_frequency[src][trg] += 1
            else:
                phrase_frequency[src][trg] = 1
        else:
            phrase_frequency[src] = {}
            phrase_frequency[src][trg] = 1

    # Normalize
    for src in phrase_frequency:
        sum_trg = sum([phrase_frequency[src][trg] for trg in phrase_frequency[src]])
        for trg in phrase_frequency[src]:
            phrase_prob.append([src,trg,str(phrase_frequency[src][trg] / sum_trg)])
    
    fp = codecs.open(fout,"w",encoding="utf-8")
    for item in phrase_prob:
        fp.writelines(" ||| ".join(item) + "\n")
    fp.close()

def build_langauge_model():
    print("Building Language Model....")
    ja, en = load_corpus()
    sentences = [sentence.strip().split() for sentence in en]
    ngram_order = 3
    train_data, vocab_data = padded_everygram_pipeline(ngram_order, sentences)
    
    trigram = {}
    bigram = {}
    unigram = {}
    for k in train_data:
        for ngram in k:
            if len(ngram) == 3:
                trigram[" ".join(ngram)] = 1
            if len(ngram) == 2:
                bigram[" ".join(ngram)] = 1
            if len(ngram) == 1:
                unigram[ngram[0]] = 1

    print("N-gram : ",len(unigram),len(bigram),len(trigram))

    train_data, vocab_data = padded_everygram_pipeline(ngram_order, sentences)

    lm = Lidstone(0.2, ngram_order)
    lm.fit(train_data, vocab_data)

    """
    logprob = log(lm.score("'m",("i",)))
    print(logprob)
    logprob = log(lm.score("'m",("i","'m")))
    print(logprob)
    """

    #Build LM model for Stack Decoder
    language_prob = defaultdict(lambda: -999.0)

    for ngram in trigram:
        item = tuple(ngram.split())
        language_prob[item] = log(lm.score(item[2],item[:2]))

    for ngram in bigram:
        item = tuple(ngram.split())
        language_prob[item] = log(lm.score(item[1],item[:1]))

    for ngram in unigram:
        item = tuple(ngram.split())
        language_prob[item] = log(lm.score(item[0]))

    return language_prob, lm

def build_stack_decoder():
    #Build Translation Model
    phrase_table = PhraseTable()
    phrase_data = codecs.open("phrase_prob.txt","r").read().splitlines()

    for phrase in phrase_data:
        src, trg, prob = phrase.split(" ||| ")
        prob = float(prob)
        phrase_table.add(tuple(src.split()), tuple(trg.split()), log(prob))

    #Build Language Model
    lmprob, lm = build_langauge_model()

    language_model = type('',(object,),{'probability_change': lambda self, context, phrase: lmprob[phrase], 'probability': lambda self, phrase: lmprob[phrase]})()

    stack_decoder = StackDecoder(phrase_table, language_model)
    return stack_decoder


def evaluate_BLEU():
    translator = build_stack_decoder()
    ja, en = load_test_corpus()

    ref = [[sent.split()] for sent in en[:TEST_MAX]]

    translation = []
    for i, sent in enumerate(ja[:TEST_MAX]):
        print(i)
        output = translator.translate(sent.split())
        print(" ".join(output))
        translation.append(output)

    chencherry =SmoothingFunction()
    BLEU = corpus_bleu(ref,translation,smoothing_function=chencherry.method1)
    print("BLEU score = ",BLEU)
    
def main_loop():
    #extract_phrase_from_corpus()
    #get_translation_probability("phrase_prob.txt")
    translator = build_stack_decoder()
    
    a = ""
    while a != "exit":
        a = input("Enter japanese text : ")
        output = translator.translate(a.split())
        print(" ".join(output))
        print(" ")

if __name__ == "__main__":
    evaluate_BLEU()