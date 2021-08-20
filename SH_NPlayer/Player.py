import math
import random


class Player(object):
    
    """ Player class details """
    
    counter = 0         # used to count instances
    
    def __init__(self, pop_size, action='C', reward=4.0, fitness=0.0, income = 4, rich = False, strategyType= 'Everything') :
        """ Create a new Player with action, reward, fitness """
        self.__income = income
        self.__rich = rich
        self.__action = action
        self.__reward = reward
        self.__rounds = 0
        self.__fitness = fitness
        self.__nextAction = None
        self.__hurt = False
        self.__strategyType = strategyType
        type(self).counter += 1    
        self.__uniqueId =  type(self).counter
    
    def __str__(self) :
         """ toString() """
         return  str(self.__uniqueId) + ': (' + str(self.__rich) + ',' \
         + str(self.__reward) + ',' + str(self.__strategyType)  + ')'
        
    def get_nextAction(self):
        return self.__nextAction
    
    def set_strategyType(self, new_strategy):
        self.__strategyType = new_strategy
        #self.__strInstance.set_strategyType(new_strategy)

    def get_strategyType(self):
        return self.__strategyType
    
    def set_fitness(self, new_fitness) :
        self.__fitness = self.__fitness + new_fitness
        
    def get_fitness(self) :
        return self.__fitness

    def get_hurt(self) :
        return self.__hurt

    def get_income(self):
        return self.__income

    def set_hurt(self, pain) :
        self.__hurt = pain
        
    def set_rounds(self, rounds):
        self.__rounds = rounds
         
    def set_reward(self, new_reward) :
        
        self.__reward = new_reward
        
    def get_reward(self) :
        return self.__reward
             
# Every time the play ground change the real action of player
# it needs to generate a new possiable next action for use 
    def set_action(self, new_action) :
        self.__action = new_action
        #self.__strInstance.set_currentAction(new_action)
        #self.set_nextAction()
        
    def get_action(self) :
        return self.__action

    @classmethod
    def PlayerInstances(cls) :
        return cls, Player.counter


