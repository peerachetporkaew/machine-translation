#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Finite State Transducer with Multiple Output

26-12-2012 - OK พอใช้ได้แล้ว
03-01-2013 - เมื่อทำ Transfer กับ [x]+ แล้วจะมีปัญหา เช่น A [x]+ C ถ้า input เป็น A B C จะทำให้ ค่า [x]+ => B C ได้ด้วย
			  Special Characters : | _ { } [ ] @ 
"""

class MultipleOutputFST:
	
	def __init__(self,name):
		self.edge = {}
		self.name = name
		self.rule = None
		self.transfer = []
		self.parent = None
		self.temp = ""
	
	def addRule(self,rule,transfer):
		self.rule = rule
		ruleL = rule.split(" ")
		current_state = self
		last_state = self
		index = 0
		while(current_state != None) and (index < len(ruleL)):
			last_state = current_state
			if ruleL[index] in current_state.edge:
				current_state = current_state.edge[ruleL[index]]
				index += 1
			else:
				current_state = None
		
		if current_state != None:
			current_state.transfer += [transfer]
			current_state.rule = rule
		else:
			for i in range(index,len(ruleL)):
				if ruleL[i][-1] != "+":
					new_state = MultipleOutputFST(ruleL[i])
					last_state.edge[ruleL[i]] = new_state
					new_state.parent = last_state
					last_state = new_state
				else: #Example : [x]+
					rule1 = ruleL[i][:-1]
					new_state = MultipleOutputFST(rule1)
					last_state.edge[rule1] = new_state
					new_state.parent = last_state
					last_state = new_state
					last_state.edge[rule1] = last_state
			last_state.transfer += [transfer]
			last_state.rule = rule
			
	def process(self,inputL,start=0):
		info = []
		path_stack = [[self,0,info]]
		path_temp = []
		finish_stack = []
		current_index = start
		loop_flag = 0
		found_surface_flag = 0
		while current_index < len(inputL):
			while path_stack != []:
				current_input = inputL[current_index]
				current,flag,info = path_stack.pop()
				#print "INDEX",current_index
				#print "CURRENT",current.name
				#print "INPUT",current_input
				found_surface_flag = 0
				new_info = info[:]
				#1.หาอันที่ตรงตัวก่อน
				if current_input in current.edge:
					#print current_input,"PASS"
					found_surface_flag = 1
					next_state = current.edge[current_input]
					next_state.temp += current_input + " "
					if next_state != current:
						new_info +=  [current_input]
					else:
						new_info[:-1] += " " + current_input
					if next_state.transfer != []:
						finish_stack += [[current_index,next_state,new_info]]
						path_temp += [[next_state,False,new_info]]
					else:
						path_temp += [[next_state,False,new_info]]
				
				new_info = info[:]
				#2.หาตัวที่ Match โดยวนหาทุก Key ที่ขึ้นต้นด้วย [
				for pattern in current.edge.keys():
					if pattern[0] == "[":
						if pattern == "[@]":
							next_state = current.edge["[@]"]
							next_state.temp += current_input + " "
							if next_state != current:
								new_info += [current_input]
							else:
								new_info[:-1] += " " + current_input
							if next_state.transfer != []:
								finish_stack += [[current_index,next_state,new_info]]
								path_temp += [[next_state,False,new_info]]
							else:
								path_temp += [[next_state,False,new_info]]
						elif "@" in current_input:
							tag = current_input.split("@")[-1]
							if tag == pattern.split("@")[1][:-1]:
								next_state = current.edge[pattern]
								next_state.temp += current_input + " "
								if next_state != current:
									new_info += [current_input]
								else:
									new_info[:-1] += " " + current_input
								if next_state.transfer != []:
									finish_stack += [[current_index,next_state,new_info]]
									path_temp += [[next_state,False,new_info]]
								else:
									path_temp += [[next_state,False,new_info]]
								
				
				#print "PATH TEMP",path_temp
			path_stack = [x for x in path_temp]
			path_temp = []
			current_index += 1

		output = finish_stack
		
		final_output = []
		
		for i in range(0,len(output)):
			for j in range(0,len(output[i][1].transfer)):
				outputT = output[i][1].transfer[j]
				for k in range(0,len(output[i][2])):
					outputT = outputT.replace("{" + str(k) + "}",output[i][2][k])
				final_output += [[output[i][0],outputT]]
		
		return final_output
			
		

if __name__ == "__main__":
	root = MultipleOutputFST("ROOT")
	root.addRule("A B+ C","ADC")
	root.addRule("A B@VBZ C","ABC")
	#root.addRule("A@X [@]+ C@X","AXC")
	root.addRule("A@X [@VBZ] C@X","C {1} A")
	root.addRule("A@X [@VBZ] C@X","CC {1} AA")
	root.addRule("C@VBZ [@X]","CX {0}")
	root.addRule("C@VBZ","CX A")
	output = root.process(["A@X","C@VBZ","C@X","D"],0)
	print(output)
	
	#ต้องเปลี่ยนมาวนทุก output แต่ละ output วนทุก Transfer
	
