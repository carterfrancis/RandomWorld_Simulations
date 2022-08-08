import numpy as np
import matplotlib.pyplot as plt
import random
import math
import matplotlib
import copy

from sklearn.utils import resample


# defining a useful function
def getFirst10(data):
    #last dimension needs to be time
    Times = np.ones([data.shape[0],data.shape[1],data.shape[2],10]) * 240
    
    for idx1 in range(data.shape[0]):
        for idx2 in range(data.shape[1]):
            for idx3 in range(data.shape[2]):
                times = np.where(data[idx1,idx2,idx3]==True)[0][:10] + 1
                Times[idx1,idx2,idx3,:times.shape[0]] = times
                Times[idx1,idx2,idx3,1:] -= Times[idx1,idx2,idx3,:-1].copy()
    return Times


class Rat:
    
    def __init__(self, startState, qvals, alpha, beta):#, gamma, epsilon):
        self.state = startState 
        self.qvals = qvals #m x n matrix; m actions, n state variables
        self.alpha = alpha #learning rate
        self.gamma = 1 #discounting is none
        self.action = float('nan')
        self.beta = beta   #learning the average reward
        self.avgR = 0
        self.selectionProbs = float('nan')
        self.rpes = []
        self.rews = []    
        self.epsilon = 0.9
            
    def update(self, rwd, newState, rwdIsRPE, update_now=True):
        '''
        This function is used at every timestep in the simulations to perform all tasks of the rat
        '''
        
        #decide action values and action
        actionVals = np.matmul(self.qvals, newState)
        aVals = actionVals - max(actionVals)
        newProbs = (math.e**aVals)/sum(math.e**aVals)
#         newProbs = (math.e**actionVals)/sum(math.e**actionVals)
        if np.random.rand()>self.epsilon:
            newAction = np.random.choice(np.arange(self.qvals.shape[0]))
        else:
            newAction = np.random.choice(np.arange(self.qvals.shape[0]), p=newProbs)  #softmax
        isExplore = False
        newQval = actionVals[newAction]
        
        #update q values
        if (not math.isnan(self.action)):# & (not self.wasExplore):  #if this is not the first move (ie. there was no earlier action)
            if rwdIsRPE:
                rpe = rwd
            else:
                qval = sum(self.qvals[self.action,:] * self.state)
                rpe = rwd - self.avgR + self.gamma*newQval - qval
                if not update_now:
                    rpe=0
            self.rpes.append(rpe)
            self.rews.append(rwd)
            self.avgR = self.avgR + self.beta*rpe
            self.qvals[self.action,:] = self.qvals[self.action,:] + self.alpha*rpe*self.state   #self.state is the gradient of the qfunction in the linear case
            
        #update state and action
        self.state = newState
        self.action = newAction
        self.selectionProbs = newProbs
        
        
class BoxWorld_basic:
     
    rest = 3 #setting other reward strengths
    sniff = 3
    groom = 3
    trialLength = 120 #number of timesteps in a trial
    
    def __init__(self,stimIsRPE,rewardStrength):
        self.time = 0
        zeroVec = np.zeros(1)
        zeroVec[0] = 1
        self.state = zeroVec
        self.stimIsRPE = stimIsRPE
        self.rewardStrength = rewardStrength
    
    def getNextState(self,action):
        '''
        This function is used at every timestep in the simulations to perform all tasks of the world 
        ie. generating states for the rat.
        '''
        
        rwdIsRPE = False
        
        if action == 3: #pressed
            reward = self.rewardStrength
            rwdIsRPE = self.stimIsRPE
        elif action == 2: #groomed
            reward = self.groom
        elif action == 1: #sniffed around
            reward = self.sniff
        elif action == 0: #rested
            reward = self.rest
        elif math.isnan(action): #first turn
            reward = 0
        
        outState = self.state
            
        return reward, outState, rwdIsRPE
    
    
class BoxWorld_complete:
    
    rest = 3
    sniff = 3
    groom = 3
    trialLength = 120
    
    def __init__(self,stimIsRPE, rewardStrengths):
        self.triad = 0
        self.trial = 0
        self.time = 0
        zeroVec = np.zeros(6)
        zeroVec[4] = 1
        self.state = zeroVec
        self.rewardStrengths = rewardStrengths
        self.stimIsRPE = stimIsRPE
    
    def getNextState(self,action):
        rwdIsRPE = False
        
        if action == 3: #pressed
            if not self.trial == 1:     #if we are in leading or trailing trial
                reward = self.rewardStrengths[self.trial]
            else:
                reward = self.rewardStrengths[2-((self.triad)%3)]
                self.state[:] = 0
                self.state[(2-((self.triad)%3))+1] = 1
                rwdIsRPE = self.stimIsRPE
        elif action == 2: #groomed
            reward = self.groom
        elif action == 1: #sniffed around
            reward = self.sniff
        elif action == 0: #rested
            reward = self.rest
        elif math.isnan(action): #first turn
            reward = 0
            
        if self.time == self.trialLength-1:
            self.time = 0
            if self.trial == 2:  #if was in trailing trial, move to leading
                self.trial = 0
                self.triad += 1
                self.state[:]=0
                self.state[4]=1
            elif self.trial == 1: #if was in test, move to trail
                self.trial += 1
                self.state[:]=0
                self.state[5]=1
            elif self.trial == 0: #if was in lead, move to test
                self.trial += 1
                self.state[:]=0
                self.state[0]=1
        else:
            self.time += 1
                
        outState = self.state
            
        return reward, outState, rwdIsRPE
    
    
#rat thinks states alternate lo hi lo hi etc
class BoxWorld_incomplete:
    
    rest = 3
    sniff = 3
    groom = 3
    trialLength = 120
    
    def __init__(self,stimIsRPE, rewardStrengths):
        self.triad = 0
        self.trial = 0
        self.time = 0
        zeroVec = np.zeros(3) #worth pressing [0,1] or not [1,0]?
        #zeroVec[0] = 1
        self.previousWorthPress = False
        self.state = zeroVec
        self.rewardStrengths = rewardStrengths
        self.stimIsRPE = stimIsRPE
    
    def getNextState(self,action):
        rwdIsRPE = False
        
        if action == 3: #pressed
            self.state[:] = 0
            if not self.trial == 1:     #if we are in leading or trailing trial
                reward = self.rewardStrengths[self.trial]
                self.state[self.trial] = 1
            else:
                reward = self.rewardStrengths[self.triad%3]
                self.state[self.triad%3] = 1
            if reward > 2:
                self.previousWorthPress = True
            else:
                self.previousWorthPress = False
            rwdIsRPE = self.stimIsRPE
        elif action == 2: #groomed
            reward = self.groom
        elif action == 1: #sniffed around
            reward = self.sniff
        elif action == 0: #rested
            reward = self.rest
        elif math.isnan(action): #first turn
            reward = 0
            
        if self.time == self.trialLength-1:
            self.time = 0
            if self.trial == 2:  #if in trailing trial
                self.trial = 0
                self.triad += 1
            else:
                self.trial += 1
            if self.previousWorthPress:
                self.state = np.array([0,0,1])
            else:
                self.state = np.array([1,0,0])
        else:
            self.time += 1
                
        outState = self.state
            
        return reward, outState, rwdIsRPE