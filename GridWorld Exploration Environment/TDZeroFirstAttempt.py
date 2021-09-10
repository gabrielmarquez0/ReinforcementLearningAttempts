import numpy as np

def updateValueFunctionTD(alpha, gamma, R, S, SPrime, V):
    V[S] = V[S] + alpha * (R + gamma * V[SPrime] - V[S])
    return V

def updateStateRewardTD(S, A, transitionTable, rewardTable):
    return (np.random.choice([state for state in transitionTable[S][A].keys()],
                             p=[prob for prob in transitionTable[S][A].values()]),
            np.random.choice([state for state in rewardTable[S][A].keys()],
                             p=[prob for prob in rewardTable[S][A].values()]))

def TD(episodesN, policy, updateValueFunction, updateStateReward, V):
    records = {}
    for episode in range(episodesN):
        S = 3
        records[episode] = {"states":[3], "valueFunction":0}
        while S > 0 and S < 6:
            A = policy(S)
            SPrime, R = updateStateReward(S,A)
            updateValueFunction(R, S, SPrime, V)
            S = SPrime
            records[episode]["states"].append(S)
        records[episode]["valueFunction"] = V.copy()
    return records
    
 
def main():
    transition = {1:{0:{0:1},1:{2:1}},
                  2:{0:{1:1},1:{3:1}},
                  3:{0:{2:1},1:{4:1}},
                  4:{0:{3:1},1:{5:1}},
                  5:{0:{4:1},1:{6:1}}
                }
    reward = {1:{0:{0:1},1:{0:1}},
              2:{0:{0:1},1:{0:1}},
              3:{0:{0:1},1:{0:1}},
              4:{0:{0:1},1:{0:1}},
              5:{0:{0:1},1:{1:1}}
             }

    
    episodesN = 1000
    policy = lambda x: np.random.randint(0,2,1)[0]

    alpha = .01
    gamma = 1
    V = {0:0,1:0.5,2:0.5,3:0.5,4:0.5,5:0.5,6:0}

    print(TD(episodesN,
       policy,
       lambda w,x,y,z: updateValueFunctionTD(alpha, gamma, w, x, y, z),
       lambda x,y: updateStateRewardTD(x,y, transition, reward),
       V))
    
if __name__ == "__main__":
    main()
