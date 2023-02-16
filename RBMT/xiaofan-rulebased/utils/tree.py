# -*- coding: utf-8 -*-
import os

class PennParsed_Tree:
	parent = -1
	value = ""
	isTerminal = 1
	children = []
	isProcess = 0
	id = -1

def parse_penn_parsed_tree2(strin):
	j = 0
	k = -1
	tree = []
	before = ""
	for i in strin:
		if i == "(":
			if j != 0:
				n.parent = j-k
				k += 1
			n = PennParsed_Tree()
			tree.append(n)
			n.id = len(tree) - 1
			j += 1
		elif i == " ":
			#n = tree[n.parent]
			n.parent = j-k
			n = PennParsed_Tree()
			tree.append(n)
			n.id = len(tree) - 1
			k += 1
			j += 1
		elif i == ")":
				if before != ")":
					k += 2
				else:
					k += 2
		else:
			n.value = n.value + i.strip()
		before = i
		print(i,k,j)
	return tree

def parse_penn_parsed_tree(strin):
	temp1 = strin.replace(" (","(")
	temp1 = temp1.replace(" "," _ ")
	temp1 = temp1.replace("("," ( ")
	temp1 = temp1.replace(")"," ) ")
	temp1 = temp1.replace("  "," ")
	temp1 = temp1.replace("  "," ")
	
	temp = temp1.strip().split(" ")
	tree = []
	n = PennParsed_Tree()
	n.value = 'root'
	n.id = 0
	tree  += [n]
	p = n
	k = -1
	flag = 0
	for i in temp:
		if i == "(":
			pass
		elif i == ")":
			p = tree[p.parent]
			if flag == 1:
				p = tree[p.parent]
				flag = 0
		elif i == "_":
			flag = 1
		else:
			n = PennParsed_Tree()
			n.value = i
			tree += [n]
			n.id = len(tree) - 1
			if p == 0:
				n.parent = -1
				p = n
			else:
				n.parent = p.id
				p = n
		#print i,len(tree),p.id
	return tree

def gen_children(tree):
	l = len(tree)
	ch = {}
	for i in range(1,l):
		if tree[i].parent == -1:
			tree[i].parent = 0
		#print tree[i].parent,i
		if tree[i].parent in ch:
			ch[tree[i].parent].append(i)
		else:
			ch[tree[i].parent] = [i]
	return ch

def convert_penn_parsed_tree_to_one_line(str_file_name_in,str_file_name_out):
	fp = open(str_file_name_in,"r").readlines()
	fo = open(str_file_name_out,"w")
	out = ""
	for line in fp:
		if line.strip() == "":
			fo.writelines(out.strip() + "\n")
			out = ""
		else:
			out += line.strip() + " "
	fo.close()

#Build complete tree from input tree and its children
def build_complete_tree(tree,children):
  l = len(tree)
  for i in range(0,l):
    tree[i].children = []
    if i in children:
      tree[i].children = [tree[k] for k in children[i]]
    tree[i].parent = tree[tree[i].parent]
    #print [m.value for m in tree[i].children]
  return tree
  
def get_penn_tree_from_string(str_input):
  tree = parse_penn_parsed_tree(str_input)
  node = gen_children(tree)
  tree = build_complete_tree(tree,node)
  return tree
  
def convert_tree_to_moses_xml(node):
  if len(node.children) == 0:
    return node.value
  else:
    output = ""
    for i in node.children:
      output += convert_tree_to_moses_xml(i)
    return "<tree label=\"" + node.value + "\">" + output + "</tree>"
  
def convert_tree_to_moses_xml_file(input_file_str,output_file_str):
  fin = open(input_file_str,"r").readlines()
  fout = open(output_file_str,"w")
  for line in fin:
    tree = get_penn_tree_from_string(line.strip())
    output = convert_tree_to_moses_xml(tree[0])
    fout.writelines(output + "\n")
  fout.close()
    

if __name__ == "__main__":
  convert_tree_to_moses_xml_file("/home/peerachet/Desktop/MT/PLAYGROUND/MT/Heirarchical/exp01/lang.src",
				  "/home/peerachet/Desktop/MT/PLAYGROUND/MT/Heirarchical/exp01/lang.src.xml")
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  