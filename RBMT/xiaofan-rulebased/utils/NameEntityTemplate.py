# -*- coding: utf-8 -*-
from .MultipleOutputFST_v06 import MultipleOutputFST

class NE:
    def __init__(self,name,type,translation):
        self.name = name
        self.type = type
        self.translation = translation

class NEDatabase:
    
    def __init__(self):
        self.level1 = {}
        
    def add(self,Ne):
        if len(Ne.name) < 3:
            self.level1[Ne.name] = Ne
            #print Ne.name
        else:
            l1 = Ne.name[-3:]
            if l1 in self.level1:
                self.level1[l1][Ne.name] = Ne
            else:
                self.level1[l1] = {}
                self.level1[l1][Ne.name] = Ne
    
    def search(self,NEname):
        if len(NEname) < 3:
            if NEname in self.level1: 
                return self.level1[NEname]
            else:
                return None
        else:
            l1 = NEname[-3:]
            if l1 in self.level1:
                if NEname in self.level1[l1]:
                    return self.level1[l1][NEname]
                else:
                    return None
            else:
                return None

def NE_Detection(SourceInput, NEDb):
    count = 0
    tokenL = SourceInput.split(" ")
    NEmappingList = {}
    templateSource = []
    for token in tokenL:
        #print "TOken = ",token
        out = NEDb.search(token)
        #print out
        if out == None:
            templateSource += [token]
        else:
            #temp = "[" + out.type + "_" + str(count) + "]"
            temp = str(count) + "@" + out.type
            templateSource += [temp]
            NEmappingList[temp] = out.translation
            count += 1
    return templateSource,NEmappingList

def NE_Template_Detection(SourceInput, NERule):
    #Left to Right Greedy Search
    SourceItem = SourceInput.split(" ")
    i = 0
    TargetItem = []
    while i < len(SourceItem):
        output = NERule.process(SourceItem,i)
        if len(output) != 0:
            #print output
            i = output[0][0]+1
            TargetItem += [output[0][1]]
        else:
            TargetItem += [SourceItem[i]]
            i += 1
    
    return " ".join(TargetItem)

def beam_search(decodingTable,LM):
    #Each State in Beam = [output,current_index,score]
    #Init Beam List
    beam = [["",0,0.0]]
    SET = []
    GOAL = []
    g = 0
    while (len(beam) > 0):
        g += 1
        
        for state in beam:
            #pass #print "State : ",state
            #Get Successor of current state
            current_index = state[1]
            successor = []
            if current_index < len(decodingTable):
                #pass #print "DI: ",decodingTable[current_index][0][0],decodingTable[current_index][0][1]
                for item in decodingTable[current_index]:
                    #pass #print "S :",state[0],state[1]
                    #pass #print "T :",item[0],item[1]
                    successor += [[state[0]+item[1],state[1]+item[0],state[2]+len(item[1])]]
            else:
                #Finish Segmentation
                #pass #print "Found :",state[0]
                #raw_input("F:")
                GOAL += [state]
                
            SET += successor
            #pass #print "SET = ",SET
            #raw_input("xx :")
            #Calculate Score on SET
            
        beam = []
        for i in range(0,len(SET)):
            #pass #print "SET i",SET[i]
            SET[i][2] = 0+SET[i][0].count("|")
        
        SET.sort(key=lambda tup: tup[2])
        SET.reverse()
        for s in SET[0:30]:
            beam += [s]
        SET = []
        pass #print "CURRENT STATE : %d, pass #print BEAM =>"%g
        #pass #print_beam(beam)
        pass #print "END BEAM"
    #print "GOAL : "
    print(GOAL)
    for i in range(0,len(GOAL)):
            #pass #print "SET i",SET[i]
            print(GOAL[i])
            GOAL[i][2] = 0+GOAL[i][0].count("|")
    GOAL.sort(key=lambda tup: tup[2])
    pass #print_beam(GOAL)
    return GOAL[-1][0]

def Translate(InputStr,TM,NED):
    t,m = NE_Detection(InputStr, NED)
    #print "T = ",t
    #print "M = ",m
    
    item = t
    table = []
    for i in range(len(item)):
        output = TM.process(t,i)
        if output == []:
            output = [[1,item[i]+" "]]
        else:
            for j in range(len(output)):
                output[j][0] -= i-1
        table += [output]
    #print table
    
    output = beam_search(table,"")
    
    #print output
    
    translation = output
    for i in m.keys():
        #print i,m[i]
        translation = translation.replace(i,m[i])
    
    print("Translation = ",translation)
    return translation

if __name__ == "__main__":
    
    Input = "Last 3 weeks I went to England with my friends ."
    
    NUM = MultipleOutputFST("NUM")
    NUM.addRule("3","3@NUM")
    output = NE_Template_Detection(Input,NUM)
    
    print("NUM = ",output)
    
    DTM = MultipleOutputFST("ROOT")
    DTM.addRule("Last [@NUM] weeks","{1}_สัปดาห์ก่อน@DTM")
    output = NE_Template_Detection(output,DTM)
    
    print("DTM = ",output)
    
    NED = NEDatabase()
    NEL = []
    NEL += [NE("England","LOC","อังกฤษ@LOC")]
    for item in NEL:
        NED.add(item)
    
    TRAN = MultipleOutputFST("ROOT")
    TRAN.addRule("[@DTM] I went to [@LOC]","ฉัน ไป {4} เมื่อ {0} ")    
    
    print("BEFORE = ",output)
    output = Translate(output,TRAN,NED)
    print(output)

