# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:36:18 2020

@author: Soheil_Pc
"""
import numpy as np
import torch
import os#for loading ai
# type_of_action = > 1 is for random .... 2 is for maxQ



#x,y is action 
#boardinit is state


class Dqn():
    
    def __init__(self,alpha2,gamma2):
        
        self.q = {}
        self.last_reward = 0
        self.last_state = np.zeros((3,3))
        self.last_action = []
        
        self.alpha = alpha2
        self.gamma = gamma2
        
    def save(self):
        
        print("saving ...")
        
        torch.save({'Q_list':self.q}, 'last_brain.pth')
        print("final Q list")
        print(self.q.values())
        
        print("saved ...")
   
    
    def load(self):
        
        print("loading...")
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.q= checkpoint['Q_list']
            print("done !")
        else:
            print("no checkpoint found...")
        
        
        
    def learn(self,new_state,new_action,reward):
        print("reward:")
        print(reward)
        print("..................................................")
        
        if tuple([tuple([tuple(new_state[0]),tuple(new_state[1]),tuple(new_state[2])]),tuple(new_action)]) in self.q.keys():
            self.q[tuple([tuple([tuple(self.last_state[0]),tuple(self.last_state[1]),tuple(self.last_state[2])]),tuple(self.last_action)])] = ((1-self.alpha)*self.q[tuple([tuple([tuple(self.last_state[0]),tuple(self.last_state[1]),tuple(self.last_state[2])]),tuple(self.last_action)])])+((self.alpha)*(reward+self.q[tuple([tuple([tuple(new_state[0]),tuple(new_state[1]),tuple(new_state[2])]),tuple(new_action)])]))
            print("here")
            print(self.q[tuple([tuple([tuple(self.last_state[0]),tuple(self.last_state[1]),tuple(self.last_state[2])]),tuple(self.last_action)])])
            print("here")
        else:
            self.q[tuple([tuple([tuple(self.last_state[0]),tuple(self.last_state[1]),tuple(self.last_state[2])]),tuple(self.last_action)])] = ((1-self.alpha)*self.q[tuple([tuple([tuple(self.last_state[0]),tuple(self.last_state[1]),tuple(self.last_state[2])]),tuple(self.last_action)])])+((self.alpha)*(self.last_reward))
            self.q[tuple([tuple([tuple(new_state[0]),tuple(new_state[1]),tuple(new_state[2])]),tuple(new_action)])]=0

        
        print("new Q list>>>>")
        self.last_reward = reward
        self.last_action = new_action
        self.last_state = new_state
        #print(self.q.values())
        print("..................................................")
        
    def update(self,board,reward):
        
        
        prob = np.random.random_sample()# random number between 0 and 1
        
        available = np.where(board == 0)
        
        if prob>0.3:
            print("it is maxQ choice ... ")
            
            available_action_1 = []
            
            
            available_zip = zip(available[0],available[1])
            
            for i in range(len(available[0])):
                available_action_1.append(available_zip.__next__())
                
            #x,y = available_action[np.argmax(list(map(lambda x:self.q[x],available_action)))]
            max_val = -1000
            available_action = np.empty(len(available_action_1),dtype=object)
            available_action[:] = available_action_1
            #print(available_action[0])
            
            for key in self.q.keys():
                if key[0] == tuple([tuple(board[0]),tuple(board[1]),tuple(board[2])]):#important
                    if self.q[key]>max_val:
                           
                        max_val = self.q[key]
                        [x,y]= key[1]
                        del available_action_1[available_action_1.index((x,y))]
                       
            for i in range(len(available_action)):
                    
                self.q[tuple([tuple([tuple(board[0]),tuple(board[1]),tuple(board[2])]),tuple(available_action[i])])] = 0
                [x,y] = available_action[i]
            
                                        
            
            
            type_of_action = 2
            
            self.last_action = (x,y)
            self.last_reward = reward
            
            self.last_state = board
            """print("last state:")
            print(self.last_state)"""
            
            return [x,y,type_of_action]
            
            
        else:
           print("it is random choice ...")
           
           number_choice = len(available[0])
           random_no = np.random.randint(0,number_choice)
           x = available[0][random_no]
           y = available[1][random_no]
           
           self.q[tuple([tuple([tuple(board[0]),tuple(board[1]),tuple(board[2])]),tuple((x,y))])] = 0
           
           self.last_action = (x,y)
           self.last_reward = reward
           self.last_state = board
           type_of_action = 1
           """print("last state in random:")
           print(self.last_state)
           print(self.last_action)"""
           return [x,y,type_of_action]







