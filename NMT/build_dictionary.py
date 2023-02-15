import sys
from collections import defaultdict

fp = open(sys.argv[1],"r",encoding="utf-8").readlines()
fo = open(sys.argv[2],"w",encoding="utf-8")
word_freq = defaultdict(int)

for line in fp:
    itemL = line.strip().split()
    for item in itemL:
        word_freq[item] += 1

for w in word_freq.keys():
    fo.writelines(w + "\t" + str(word_freq[w]) + "\n")

fo.close()