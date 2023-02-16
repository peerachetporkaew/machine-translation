from utils.MultipleOutputFST_v06 import *
from utils.NameEntityTemplate import *
import codecs 
def load_dictionary(fname):
    fp = codecs.open(fname,encoding="utf-8").readlines()
    dictItem = []
    for i in range(0,len(fp),5):
        dictItem.append((fp[i].strip(),fp[i+1].strip()+" "))
    return dictItem 

class PhraseTranslator():

    def __init__(self,modelfile):

        NED = NEDatabase()
        NEL = []
        NEL += [NE("อังกฤษ","LOC","England@LOC")]
        NEL += [NE("football","SPORT","กีฬาฟุตบอล")]
        NEL += [NE("Marry","PERSON","แมรี่")]

        for item in NEL:
            NED.add(item)

        self.NED = NED
        self.root = MultipleOutputFST("ROOT")

        dictItem = load_dictionary(modelfile)
        for k in dictItem:
            self.root.addRule(k[0],k[1] + ""*(len(k[0].split())-1)*2)


    def process(self,wordList,start):
        return self.root.process(wordList,start)

    def translate(self,sourceStr):
        return Translate(sourceStr,self.root,self.NED)


if __name__ == "__main__":
    translator = PhraseTranslator("dictionary/basic.txt")
    translator.translate("I love Marry .")
    translator.translate("I love football .")
