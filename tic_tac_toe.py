# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:13:42 2020

@author: Soheil_Pc
"""
import numpy as np

from ai_tic import Dqn
brain = Dqn(0.1,0.9)

winner = 0 # 1 is first,2 is ai , 0 is draw
turn = 1#1 is first player,2 is ai
board = np.zeros((3,3))#board is state here
last_action =[]
last_reward = 0
ai_win_no = 0
draw_no = 0

def check_winner():
    global board
    global winner 
    global ai_win_no
    global draw_no
    
    for i in range(3):
        if np.all((board[:,i])==[1,1,1]) or np.all((board[i])==[1,1,1]):
            winner = 1
            return winner
        elif np.all((board[:,i])==[2,2,2]) or np.all((board[i])==[2,2,2]):
            winner = 2
            ai_win_no = ai_win_no + 1
    if np.all(([board[0,0],board[1,1],board[2,2]])==[1,1,1]):
        winner = 1
    if np.all(([board[0,0],board[1,1],board[2,2]])==[2,2,2]):
        winner = 2
        ai_win_no = ai_win_no + 1
    if winner == 0 and len(np.where(board==0)[0])==0:
        winner = 3#draw
        draw_no = draw_no + 1
            
def terminate():
    global board
    global winner
    global last_reward
    global turn
    global last_action
    global brain
    
    brain.learn(np.copy(board),np.copy(last_action),np.copy(last_reward))
    last_reward = 0
    board = np.zeros((3,3))
    turn = 1
    winner = 0
                

def main():
    global turn 
    global board
    global last_reward
    global winner
    global last_action
    global brain
    global ai_win_no
    global draw_no
    
    brain.load()
    i = 0
    while i<100:
        if turn == 1:
            print("turn of first player.....")
            #[x,y]=int(input()),int(input())
            available = np.where(board == 0)##
            
            number_choice = len(available[0])##
            random_no = np.random.randint(0,number_choice)##
            x = available[0][random_no]##
            y = available[1][random_no]##
            
            flag = 1
            # while flag:
            #     if board[x][y]==0:
            #         flag = 0
            #         print("correct...")
            #         """number_choice = len(available[0])
            #         random_no = np.random.randint(0,number_choice)
            #         x = available[0][random_no]
            #         y = available[1][random_no]"""
            #     else:
            #         print("try again ....")
                    
            #         [x,y]=int(input()),int(input())
                    
            board[x][y]=1
            turn = 2
            check_winner()
            if winner == 1 or winner==2:
                print("winner:"+str(winner))
                last_reward = -2
                        
                terminate()
                i = i + 1
                
                
            
            elif winner == 3:
                last_reward = -0.5
                terminate()
                i = i + 1
                print("---------------------draw------------------------")
            print('board:'+str(board))   
        else:
            print("turn of Qmax player .....")
            
            
            [x,y,type_random]= brain.update(np.copy(board),np.copy(last_reward))
            
            
            board[x][y] = 2
            
            last_action = (x,y)
            turn = 1
            check_winner()
            if winner != 0 :
                if winner == 3:
                    last_reward = -0.5
                    terminate()
                    i = i + 1
                    print("---------------------draw------------------------")
                else:   
                    print("winner :"+str(winner))
                    last_reward = 2
                    terminate()
                    i = i + 1            
            else:
                last_reward = 0
                
                brain.learn(np.copy(board),np.copy(last_action),np.copy(last_reward))
            print('board:'+str(board))
    brain.save()

    print("finished....") 
    print("ai win numbers :")
    print(ai_win_no)  
    print("draw no")
    print(draw_no)
    
# Running the whole thing
if __name__ == '__main__':
    main().run()
        
        
#28 win .. 15 draw----34 _ 18 ----32 __ 17---35_27
#with gamma = 0.5 >> 336......328...<>...with gamma=0.9  >>  342 ... 324
        
        
