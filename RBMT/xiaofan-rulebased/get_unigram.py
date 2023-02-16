from nltk import ngrams, FreqDist
all_counts = dict()

data = open("cut_result.txt","r").read()


all_counts[1] = FreqDist(ngrams(data.split(), 1))
for item in all_counts[1]:
    print(item[0],all_counts[1][item])