
# Attribution Information: This code was developed off work done by Michael Kirley

class Player(object):
    
    """ Player class details """
    

    
    def __init__(self, pop_size, action='C', reward=4.0, id=1, income = 4.0, rich = False, moral=0.0, strategyType= 'Everything') :
        """ Create a new Player with action, reward, fitness """
        self.__income = income
        self.__rich = rich
        self.__action = action
        self.__reward = reward
        self.__rounds = 0
        self.__nextAction = None
        self.__hurt = False
        self.__strategyType = strategyType
        self.__moralFactor = moral
        self.__previousContribution = 0
        self.previous_reward = 0
        self.__uniqueId = id
    
    def __str__(self) :
         """ toString() """
         return  str(self.__uniqueId) + ': (' + str(self.__rich) + ',' \
         + str(self.__reward) + ',' + str(self.__strategyType)  + ')'
        
    def get_nextAction(self):
        return self.__nextAction
    
    def set_strategyType(self, new_strategy):
        self.set_hurt(False)
        if isinstance(new_strategy, str):
            self.__strategyType = new_strategy
        else:
            self.__strategyType = new_strategy[0]

    def get_strategyType(self):
        return self.__strategyType

    def set_previous_reward(self, reward):
        self.previous_reward = reward

    def get_previous_reward(self):
        return self.previous_reward

    def get_hurt(self) :
        return self.__hurt

    def get_income(self):
        return self.__income

    def set_hurt(self, pain) :
        self.__hurt = pain

    def set_previous_Contribution(self, contribution):
        self.__previousContribution = contribution

    def set_rounds(self, rounds):
        self.__rounds = rounds
         
    def set_reward(self, new_reward) :
        
        self.__reward = new_reward
        
    def get_reward(self) :
        return self.__reward

    def get_rich(self):
        return self.__rich

    def get_morals(self):
        return self.__moralFactor

    def get_moral_factor(self, target):
        contributed = self.__previousContribution/target
        if contributed < 1 and contributed < self.__moralFactor:
            return self.__moralFactor - contributed
        else:
            return 0
        
    def get_action(self) :
        return self.__action

    @classmethod
    def PlayerInstances(cls) :
        return cls, Player.counter


