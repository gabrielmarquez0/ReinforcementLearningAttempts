import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def updateValueFunctionQLearning(alpha, gamma, S, A, R, SPrime, Q):
    maxValue = max(Q[SPrime].values())
    Q[S][A] = Q[S][A] + alpha * (R + gamma * maxValue - Q[S][A])
    return Q

def updateStateQLearning(S, A, transitionTable):
    return np.random.choice([state for state in transitionTable[S][A].keys()],
                              p=[prob for prob in transitionTable[S][A].values()])
            
def updateRewardQLearning(S, A, rewardTable):
    return  np.random.choice([reward for reward in rewardTable[S][A].keys()],
                             p=[prob for prob in rewardTable[S][A].values()])

def eGreedy(epsilon, S, Q):
    greedy = np.random.choice([True, False],p = [1-epsilon, epsilon])
    possibleActions = []
    if greedy:
        possibleActions = [action for action, value in Q[S].items()
                           if Q[S][action] == max([value for value in Q[S].values()])]
    else:
        possibleActions = [action for action in Q[S].keys()]
    return np.random.choice(possibleActions)
        

def QLearning(episodesN, startState, endState, policy, updateValueFunction, updateState, updateReward, Q):
    records = {}
    for episode in range(episodesN):
        S = startState
        records[episode] = {"states": [], "actions": []}
        while S != endState:
            A = policy(S, Q)
            SPrime = updateState(S, A)
            R = updateReward(S, A)
            updateValueFunction(S, A, R, SPrime, Q)
            records[episode]["states"].append(S)
            records[episode]["actions"].append(A)
            S = SPrime
        records[episode]["valueFunction"] = Q.copy()
    return records

#only works for deterministic transitions
#include probabilistic transitions, and maybe a percentage error calculator
def recommendedRoute(Q, policy, updateState, startState, endState):
    movement = []
    state = startState
    while state != endState and not(state in movement):
        movement.append(state)
        action = policy(state, Q)
        state = updateState(state, action)
        print(state)
        print(endState)
        print(state in movement)
    return movement

def valueFunctionConvergence(Q1, Q2, decimalTolerance):
    #(Q1.keys(), Q1.values().keys(), Q1.values().values()
    return 0

 
def main():
    #cliff gridworld, deterministic moves case, no wall bounces
    transition = {0:{0:{0:1},1:{100:1}},
                           1:{1:{110:1},2:{0:1}},
                           100:{0:{101:1},1:{200:1},3:{0:1}},
                           101:{0:{102:1},1:{201:1},2:{100:1},3:{0:1}},
                           102:{0:{103:1},1:{202:1},2:{101:1},3:{0:1}},
                           103:{0:{104:1},1:{203:1},2:{102:1},3:{0:1}},
                           104:{0:{105:1},1:{204:1},2:{103:1},3:{0:1}},
                           105:{0:{106:1},1:{205:1},2:{104:1},3:{0:1}},
                           106:{0:{107:1},1:{206:1},2:{105:1},3:{0:1}},
                           107:{0:{108:1},1:{207:1},2:{106:1},3:{0:1}},
                           108:{0:{109:1},1:{208:1},2:{107:1},3:{0:1}},
                           109:{0:{110:1},1:{209:1},2:{108:1},3:{0:1}},
                           110:{1:{210:1},2:{109:1},3:{1:1}},
                           200:{0:{201:1},1:{300:1},3:{100:1}},
                           201:{0:{202:1},1:{301:1},2:{200:1},3:{101:1}},
                           202:{0:{203:1},1:{302:1},2:{201:1},3:{102:1}},
                           203:{0:{204:1},1:{303:1},2:{202:1},3:{103:1}},
                           204:{0:{205:1},1:{304:1},2:{203:1},3:{104:1}},
                           205:{0:{206:1},1:{305:1},2:{204:1},3:{105:1}},
                           206:{0:{207:1},1:{306:1},2:{205:1},3:{106:1}},
                           207:{0:{208:1},1:{307:1},2:{206:1},3:{107:1}},
                           208:{0:{209:1},1:{308:1},2:{207:1},3:{108:1}},
                           209:{0:{210:1},1:{309:1},2:{208:1},3:{109:1}},
                           210:{1:{310:1},2:{309:1},3:{110:1}},
                           300:{0:{301:1},3:{200:1}},
                           301:{0:{302:1},2:{300:1},3:{201:1}},
                           302:{0:{303:1},2:{301:1},3:{202:1}},
                           303:{0:{304:1},2:{302:1},3:{203:1}},
                           304:{0:{305:1},2:{303:1},3:{204:1}},
                           305:{0:{306:1},2:{304:1},3:{205:1}},
                           306:{0:{307:1},2:{305:1},3:{206:1}},
                           307:{0:{308:1},2:{306:1},3:{207:1}},
                           308:{0:{309:1},2:{307:1},3:{208:1}},
                           309:{0:{310:1},2:{308:1},3:{209:1}},
                           310:{2:{309:1},3:{210:1}}
                           }
                            
    #cliff (-1 for all states, -100 for moving onto cliff) deterministic
    reward = {0:{0:{-100:1},1:{-1:1}},
                   1:{1:{-1:1},2:{-100:1}},
                   100:{0:{-1:1},1:{-1:1},3:{-1:1}},
                   101:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   102:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   103:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   104:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   105:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   106:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   107:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   108:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   109:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-100:1}},
                   110:{1:{-1:1},2:{-1:1},3:{-1:1}},
                   200:{0:{-1:1},1:{-1:1},3:{-1:1}},
                   201:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   202:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   203:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   204:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   205:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   206:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   207:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   208:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   209:{0:{-1:1},1:{-1:1},2:{-1:1},3:{-1:1}},
                   210:{1:{-1:1},2:{-1:1},3:{-1:1}},
                   300:{0:{-1:1},3:{-1:1}},
                   301:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   302:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   303:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   304:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   305:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   306:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   307:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   308:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   309:{0:{-1:1},2:{-1:1},3:{-1:1}},
                   310:{2:{-1:1},3:{-1:1}}
                   }

    #Q(S,A) given no bounce
    Q = {0:{0:0,1:0},
                   1:{1:0,2:0},
                   100:{0:0,1:0,3:0},
                   101:{0:0,1:0,2:0,3:0},
                   102:{0:0,1:0,2:0,3:0},
                   103:{0:0,1:0,2:0,3:0},
                   104:{0:0,1:0,2:0,3:0},
                   105:{0:0,1:0,2:0,3:0},
                   106:{0:0,1:0,2:0,3:0},
                   107:{0:0,1:0,2:0,3:0},
                   108:{0:0,1:0,2:0,3:0},
                   109:{0:0,1:0,2:0,3:0},
                   110:{1:0,2:0,3:0},
                   200:{0:0,1:0,3:0},
                   201:{0:0,1:0,2:0,3:0},
                   202:{0:0,1:0,2:0,3:0},
                   203:{0:0,1:0,2:0,3:0},
                   204:{0:0,1:0,2:0,3:0},
                   205:{0:0,1:0,2:0,3:0},
                   206:{0:0,1:0,2:0,3:0},
                   207:{0:0,1:0,2:0,3:0},
                   208:{0:0,1:0,2:0,3:0},
                   209:{0:0,1:0,2:0,3:0},
                   210:{1:0,2:0,3:0},
                   300:{0:0,3:0},
                   301:{0:0,2:0,3:0},
                   302:{0:0,2:0,3:0},
                   303:{0:0,2:0,3:0},
                   304:{0:0,2:0,3:0},
                   305:{0:0,2:0,3:0},
                   306:{0:0,2:0,3:0},
                   307:{0:0,2:0,3:0},
                   308:{0:0,2:0,3:0},
                   309:{0:0,2:0,3:0},
                   310:{2:0,3:0}
                   }


    episodesN = 200
    alpha = .5
    gamma = 1
    epsilon = 0.1
    startState = 0
    endState = 1
    records = QLearning(episodesN,
       startState,
       endState,
       lambda x,y: eGreedy(epsilon, x, y),
       lambda v,w,x,y,z: updateValueFunctionQLearning(alpha, gamma, v, w, x, y, z),
       lambda x,y: updateStateQLearning(x,y, transition),
       lambda x,y: updateRewardQLearning(x,y, reward),
       Q)

    print(records)
    
if __name__ == "__main__":
    main()
