from nltk.parse import CoreNLPParser
from utils.tree import *
import jieba
from phrase_translator import PhraseTranslator
from utils.MultipleOutputFST_v06 import *

parser = CoreNLPParser('http://localhost:59000')

reorderer = PhraseTranslator("./dictionary/reorder.txt")
translator = PhraseTranslator("./dictionary/basic.txt")

def parse_chinese(inputText):
    
    #out = parser.parse(parser.tokenize(u'%s'%inputText))
    out = jieba.cut(inputText, cut_all=False)
    print(out)
    out = parser.parse(out)

    root = list(out)[0]
    root.pretty_print()
    output = root.pformat()
    return output.replace("\n","")

def get_children_list(node):
    out = []
    for ch in node.children:
        out.append(ch.value)
    return " ".join(out)

def get_leaves(node, output):
    if len(node.children) > 0:
        for ch in node.children:
            get_leaves(ch,output)
    else:
        output.append(node.value)


def get_all_leaves(node):
    output = []
    get_leaves(node,output)
    return output

def find_children(node,value):
    if node.value == value:
        return node
    else:
        found = None
        for ch in node.children:
            found = find_children(ch,value)
            if found is not None:
                return found 
        
        return None

def apply_reordering(node):
    #print("NODE",node.value, node.parent.value)
    children = get_children_list(node)
    #print("CHILDRENx",children)

    if len(node.children) == 0:
        return 

    #if len(node.children) == 1:
    #    apply_reordering(node.children[0])
    #    return 

    if node.value == "NP":
        #children = get_children_list(node)
        #print("CHILDREN",children)
        if len(node.children) == 2:
            children = get_children_list(node)
            if children == "DNP NP": # ..的 NP --> NP ..的
                #print("FOUND")
                node.children = [node.children[1], node.children[0]]
            
            if children == "QP NP": # QP (两 只) NP (狗)
                node.children = [node.children[1], node.children[0]]
                noun = get_all_leaves(node.children[0])[-1]
                #print(noun)
                nodex = find_children(node.children[1],"M")
                if nodex is not None:
                    cl = nodex.children[0].value
                    nodex.children[0].value = f"$CL({cl},{noun})"

            if children == "ADJP NP": # 白色 自行车
                node.children = [node.children[1], node.children[0]]

            if children == "DP NP": # 那 自行车
                node.children = [node.children[1], node.children[0]]
            
            if children == "PN NN": # 我們 自行车
                node.children = [node.children[1], node.children[0]]

            if children == "NP NP": # 白色 自行车
                node.children = [node.children[1], node.children[0]]
            
            if children == "NN NN": # 香港 自行车
                node.children = [node.children[1], node.children[0]]
            

        elif len(node.children) == 3:
            children = get_children_list(node)
            if children == "QP ADJP NP": # 我有两百只白狗。
                node.children = [node.children[2], node.children[1], node.children[0]]
                noun = get_all_leaves(node.children[0])[-1]
                nodex = find_children(node.children[2],"M")
                if nodex is not None:
                    cl = nodex.children[0].value
                    #print(cl)
                    nodex.children[0].value = f"$CL({cl},{noun})"
            
            elif children == "DNP ADJP NP": # 我 的白色自行车。
                node.children = [node.children[2], node.children[1], node.children[0]]

            elif children == "QP DNP NP":
                print("XX")
                node.children = [node.children[2], node.children[1], node.children[0]]
                noun = get_all_leaves(node.children[0])[-1]
                print(noun)
                nodex = find_children(node.children[2],"M")
                print(nodex)
                if nodex is not None:
                    cl = nodex.children[0].value
                    #print(cl)
                    nodex.children[0].value = f"$CL({cl},{noun})"

        elif len(node.children) == 4:
            children = get_children_list(node)
            #print(children)
            if children == "DNP QP ADJP NP": # 我 的两百只白狗。
                #print(children)
                node.children = [node.children[3], node.children[2], node.children[1], node.children[0]]
                noun = get_all_leaves(node.children[0])[-1]
                #print(noun)
                nodex = find_children(node.children[2],"M")
                cl = nodex.children[0].value
                nodex.children[0].value = f"$CL({cl},{noun})"
                

    elif node.value == "DNP":  # DNP (NP (..) DEG (的) )
        if len(node.children) == 2:
            children = get_children_list(node)
            if children == "NP DEG": # NN 的 --> 的 NN
                node.children = [node.children[1], node.children[0]]
                node.children[0].children[0].value += "/OF"
            elif children == "ADJP DEG": # NN 的 --> 的 NN
                node.children = [node.children[0], node.children[1]]
                node.children[1].children[0].value += "/DEL"

    elif node.value == "VP":
        if len(node.children) == 2:
            children = get_children_list(node)
            if children == "ADVP VP": # 很  高超
                node.children = [node.children[1], node.children[0]]
            elif children == "PP VP":
                node.children = [node.children[1], node.children[0]]

        if len(node.children) == 3:
            children = get_children_list(node)
            if children == "VV DER VP": # 跳 得 很慢
                node.children[1].children[0].value = "得/DEL"

    elif node.value == "LCP":
        if len(node.children) == 2:
            children = get_children_list(node)
            if children == "NP LC": # 在 桌子 上
                print("P")
                node.children = [node.children[1], node.children[0]]
                node.children[0].children[0].value = node.children[0].children[0].value + "@P"
    
    elif node.value == "IP":
        if len(node.children) == 4:
            children = get_children_list(node)
            if children == "DP NP VP PU": # 这些 货物 定价 过高 。
                node.children = [node.children[1], node.children[0], node.children[2], node.children[3]]

    
    for ch in node.children:
        apply_reordering(ch)
        #print("--")

    return


def process_reordering(inputText):
    out = parse_chinese(inputText)
    print(out)
    y = get_penn_tree_from_string(out)
    print(y[0].children[0].value)

    apply_reordering(y[0].children[0])

    output = []
    get_leaves(y[0].children[0],output)
    print(output)


    return " ".join(output)

def translate(inputStr):
    out = process_reordering(inputStr) # Analysis and Transfer
    print("Transfer : ",out)
    
    trans = translator.translate(out)
    print(trans)

def translate_V2(inputStr):
    out = jieba.cut(inputStr, cut_all=False)
    out = " ".join(out)
    print(out)
    out = reorderer.translate(out)
    print("REORDER : ",out)
    out = reorderer.translate(out)
    print("REORDER : ",out)
    trans = translator.translate(out)

if __name__ == "__main__":
    parse_chinese("我有两百只狗。")
    
    #translate_V2("我有两百只狗。")

    #process_reordering("我的两百只白狗。")
    #process_reordering("我的自行车是白色的。")
    #process_reordering("这星期他会很忙。")
    #process_reordering("我的日语相当差。")
    #process_reordering("她匆匆赶往机场。")
    #process_reordering("他们衷心欢迎他。")
    #process_reordering("来看看伦敦的名胜。")
    #process_reordering("实际上我是专程来看你的。") #XXXXX
    #process_reordering("请把我列入名单中。") #xxxx
    #process_reordering("我的猫在桌子上。")
    #process_reordering("过了一会儿，我们在一些白杨树下面找到了一个遮阴的地方。")
    #process_reordering("火车及时到达了。")
    #process_reordering("坐在你妹妹旁边。")
    #process_reordering("我懂一点儿德语。")
    #process_reordering("你想要些番茄汁吗？")
    #process_reordering("许多灯照亮了街道。")
    #process_reordering("这些货物定价过高。")
    #process_reordering("莱茵河发源于何处？")
    #translate("我有两百只狗。")
    