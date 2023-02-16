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
            if self.level1.has_key(l1):
                self.level1[l1][Ne.name] = Ne
            else:
                self.level1[l1] = {}
                self.level1[l1][Ne.name] = Ne
    
    def search(self,NEname):
        if len(NEname) < 3:
            if self.level1.has_key(NEname): 
                return self.level1[NEname]
            else:
                return None
        else:
            l1 = NEname[-3:]
            if self.level1.has_key(l1):
                if self.level1[l1].has_key(NEname):
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
            SET[i][2] = 1-(SET[i][0].count("|")*1.0 / len(SET[i][0]))  
        
        SET.sort(key=lambda tup: tup[2])
        SET.reverse()
        for s in SET[0:30]:
            beam += [s]
        SET = []
        pass #print "CURRENT STATE : %d, pass #print BEAM =>"%g
        #pass #print_beam(beam)
        pass #print "END BEAM"
    pass #print "GOAL : "
    
    for i in range(0,len(GOAL)):
            #pass #print "SET i",SET[i]
            GOAL[i][2] = 1-(GOAL[i][0].count("|")*1.0 / len(GOAL[i][0]))
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
    
    #print "Translation = ",translation
    return translation

def LoopTemplate(Inp,PATTERN):
    x = Inp
    output = ""
    while output != x:
        output = x
        x = NE_Template_Detection(output,PATTERN)
    return output

def ConvertDIGITLToNumber(Inp):
    print("CONVERT",Inp)
    Token = Inp.replace("(","").replace(")","").replace("_"," ").strip().split()
    #print Token
    output = []
    for t in Token:
        print(t)
        if t == "":
            continue
        if t == "@DIGITM":
            a = output.pop()
            b = output.pop()
            output += [a*b]
        elif t == "@DIGITS":
            a = output.pop()
            b = output.pop()
            print(a,b)
            output += [10+a]
        elif t == "@DIGITP":
            a = output.pop()
            b = output.pop()
            output += [a+b]
        elif t == "@DIGITL":
            output = [sum(output)]
        else:
            temp = t.split("@")[0]
            output += [int(temp)]
        print(output)
    return output.pop()
    
def TranslateNumber(Inp):
    Token = Inp.split(" ")
    output = []
    for t in Token:
        if t.endswith("@DIGITL"):
            output += [str(ConvertDIGITLToNumber(t))]
        else:
            output += [t]
    return " ".join(output)

def isNumber(Inp):
    try:
        x = int(Inp)
        return True
    except:
        return False

def TagNumber(Inp):
    token = Inp.split(" ")
    output = []
    for t in token:
        if isNumber(t):
            output += [t + "@NUM"]
        else:
            output += [t]
    return " ".join(output)

def ParseThaiNumber(input):
    
    Input = "เขา มี เงิน สาม พัน หนึ่ง ร้อย ห้า สิบ หก บาท"
    Input = "เขา มี เงิน สาม พัน หก สิบ บาท สอง หนึ่ง สาม ห้า แปด สิบ"
    Input = "ของ สอง ชิ้น นี้ ราคา เท่า กัน คือ สาม พัน หก ร้อย สิบ บาท"
    Input = "วัน ที่ ยี่ สิบ สาม เดือน เจ็ด ปี สอง พัน สิบ สี่"
    Input = "เขา มี เงิน สาม หมื่น บาท แลก เป็น เงิน ได้ หนึ่ง ร้อย ดอลล่าร์"
    Input = "เขา มี เงิน สาม หมื่น ล้าน บาท แลก เป็น เงิน ได้ หนึ่ง ร้อย ดอลล่าร์"
    #Input = "สอง แสน แปด หมื่น สิบ ห้า ล้าน บาท"
    #Input = "สินค้า ราคา สอง ล้าน แปด แสน ยี่ สิบ ห้า บาท"
    
    Input = input

    NUM = MultipleOutputFST("NUM")
    NUM.addRule("หนึ่ง","1@NUM")
    NUM.addRule("เอ็ด","1@NUM")
    NUM.addRule("ยี่","2@NUM")
    NUM.addRule("สอง","2@NUM")
    NUM.addRule("สาม","3@NUM")
    NUM.addRule("สี่","4@NUM")
    NUM.addRule("ห้า","5@NUM")
    NUM.addRule("หก","6@NUM")
    NUM.addRule("เจ็ด","7@NUM")
    NUM.addRule("แปด","8@NUM")
    NUM.addRule("เก้า","9@NUM")
    NUM.addRule("สิบ","10@SIP")
    NUM.addRule("ล้าน","1000000@LAAN")
    
    DIGIT = MultipleOutputFST("DIGIT")
    DIGIT.addRule("สิบ","10@DIGIT")
    DIGIT.addRule("ร้อย","100@DIGIT")
    DIGIT.addRule("พัน","1000@DIGIT")
    DIGIT.addRule("หมื่น","10000@DIGIT")
    DIGIT.addRule("แสน","100000@DIGIT")
    
    NUML1 = MultipleOutputFST("DNUM")
    NUML1.addRule("[@NUM] [@DIGIT]","(_{0}_{1}_)_@DIGITM")
    NUML1.addRule("[@NUM] [@DIGIT]","(_{0}_{1}_)_@DIGITM")
    NUML1.addRule("[@NUM] [@SIP]","(_{0}_{1}_)_@DIGITM")
    
    
    
    NUML2 = MultipleOutputFST("DNUM2")
    NUML1.addRule("[@SIP] [@NUM]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGITM] [@DIGITM]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGITM] [@NUM]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGITM] [@SIP]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGIT] [@DIGITS]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGITP] [@NUM]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGIT] [@DIGITP]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGIT] [@DIGITM]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGIT] [@NUM]","(_{0}_{1}_)_@DIGITP")
    NUML2.addRule("[@DIGITP] [@SIP]","(_{0}_{1}_)_@DIGITP")
    
    NUML3 = MultipleOutputFST("DNUM3")
    NUML3.addRule("[@NUM]","(_{0}_)_@DIGITL")
    NUML3.addRule("[@DIGITP]","(_{0}_)_@DIGITL")
    NUML3.addRule("[@DIGITM]","(_{0}_)_@DIGITL")
    NUML3.addRule("[@DIGITS]","(_{0}_)_@DIGITL")
    
    NUML4 = MultipleOutputFST("LAAN")
    NUML4.addRule("[@DIGITL] [@LAAN] [@DIGITL]","(_(_(_{0}_{1}_)_@DIGITM_{2}_)_@DIGITP_)_@DIGITL")
    NUML4.addRule("[@DIGITL] [@LAAN]","(_{0}_{1}_)_@DIGITM_)_@DIGITL")
    
    NUML5 = MultipleOutputFST("LAAN_SUM")
    NUML5.addRule("[@DIGITL] [@DIGITL]","(_{0}_{1}_)_@DIGITL")
    
    output = LoopTemplate(Input,NUM)
    
    #print(output)
    output = LoopTemplate(output,DIGIT)
    #print(output)
    
    output = LoopTemplate(output,NUML1)
    output = LoopTemplate(output,NUML2)
    output = LoopTemplate(output,NUML3)
    output = LoopTemplate(output,NUML4)
    output = LoopTemplate(output,NUML5)
    #print "DTM = ",output
    #raw_input()
    
    output = TranslateNumber(output)
    #print "OUT = ",output
    
    #raw_input()
    output = TagNumber(output)
    #print "NUM = ",output
    
    CUR = MultipleOutputFST("CUR")
    CUR.addRule("บาท","บาท@CUR")
    CUR.addRule("สตางค์","สตางค์@CUR")
    CUR.addRule("เยน","เยน@CUR")
    CUR.addRule("ดอล ลาร์","ดอลลาร์@CUR")
    CUR.addRule("ปอนด์","ปอนด์@CUR")
    CUR.addRule("หยวน","หยวน@CUR")
    
    CURTag = MultipleOutputFST("CURTag")
    CURTag.addRule("[@NUM] [@CUR]","({0}_{1})@CURTag")
    
    output = LoopTemplate(output,CUR)
    output = LoopTemplate(output,CURTag)

    return output
    
    

if __name__ == "__main__":
    x = ParseThaiNumber("พัน ยี่ สิบ สี่")
    print(x)
