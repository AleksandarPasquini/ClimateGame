# coding=UTF-8

from __future__ import division
from Player import Player
from random import shuffle
import random
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import sys
import math
import time
import csv
import os,errno
import os.path
import shutil
import numpy as np
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
rank = 1
mpi_processes = 1

class game(object):
    def __init__(self, pop_size, pop_update_rule, pop_update_rate, F, cost, M, group_size, target, riskType):
        self.pop_size = pop_size
        self.pop_update_rule = pop_update_rule
        self.pop_update_rate = pop_update_rate
        self.F = F
        self.cost = cost
        self.M = M
        self.group_size = group_size
        self.target = target
        self.population = []
        self.cooperators = []
        self.defectors = []
        self.lst = []
        self.avReward = 0
        self.cooperator_count = 0
        self.defector_count = 0
        self.riskType = riskType
        self.probability_of_punishment = 0
        self.punishment_amount = 0.5
        self.previous_average = 0
        self.rounds = 10

    def set_pop_update_rule(self, new_update_rule):
        self.pop_update_rule = new_update_rule
        
    def reset_cooperators(self):
        a = self.cooperators[0]
        print (a)
        self.cooperators = [a]

    def get_cooperators(self):
        return self.cooperators

    def reset_defectors(self):
        a = self.__defectors[0]
        self.defectors = [a]
        
    def get_defectors(self):
        return self.defectors

    def get_population(self):
        return self.population

    def csvDataBuilder(self, run_number, round):
        lst = self.lst
        lste = [run_number, self.pop_size, self.pop_update_rule, self.pop_update_rate, self.F, self.cost, self.M, self.group_size, self.mutation_rate,  self.cooperators[round], self.defectors[round], self.avReward, round]
        lst.append(lste)
    

    def get_lst(self):
        return self.lst

    def reset_lst(self):
        m = self.lst[0]
        self.lst = [m]

    # calculate the average reward of each round.
    def average_reward(self):
        population = self.population
        pop_size = self.pop_size
        totalreward = 0
        for player in population:
            totalreward = totalreward + player.get_reward()
        self.avReward = totalreward/pop_size
        
    def init_population(self):
        population = []
        strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
        rich_group = []
        poor_group = []
        pop_size = self.pop_size
        pop_update_rule = self.pop_update_rule
        group_size = self.group_size
        cooperator_count = 0
        defector_count = 0
        for r in range(0,pop_size) :
            if random.random() <= 0.5:
                player = Player(pop_size, None, reward=4.0, fitness=0.0, income = 4, rich = True, strategyType= np.random.choice(strategies, 1))
                rich_group.append(player)
            else :                                
                player = Player(pop_size, None, reward=2.0, fitness=0.0, income = 2, rich = False, strategyType= np.random.choice(strategies, 1))
                poor_group.append(player)
            population.append(player)
        cooperator_proportion = (cooperator_count / pop_size)
        self.population = population
        self.cooperator_count = cooperator_count
        self.defector_count = defector_count
        self.cooperators.append(cooperator_count / pop_size)
        self.defectors.append(defector_count / pop_size)
        groups = [rich_group, poor_group]
        return groups

    def risk_function(self, x):
        if self.riskType == "Linear":
            return 1-x
        if self.riskType == "Low Probability Power function":
            return 1-(x**0.1)
        if self.riskType == "High Probability Power function":
            return 1-(x**10)
        if self.riskType == "Curves exhibiting threshold effects":
            return 1 / (math.e ** (10 * (x - 1 / 2)) + 1)

    # Calculate the reward of players in a group
    def play_game(self, groups):
        # F = self.F
        # cost = self.cost
        # M = self.M
        # group_size = len(group)
        # cooperator_count = 0
        # for i, player in enumerate(group) :
        #     if player.get_action() == 'C' :
        #         cooperator_count = cooperator_count + 1
        # sigma = 0
        # delta_count = cooperator_count - M
        #
        # if delta_count < 0:           # Heaviside function
        #     sigma = 0
        # else:
        #     sigma = 1
        #
        # payD = (cooperator_count * F / group_size * cost) * sigma
        # payC = payD - cost
        group = groups[0]+groups[1]
        target = self.target
        for x in range(self.rounds):
            for player in group:
                if player.get_strategyType() == "Nothing":
                    continue
                if player.get_strategyType() == "Everything":
                    target -= player.get_reward()
                    player.set_reward(0)
                if player.get_strategyType() == "FairShare":
                    target -= player.get_reward()/self.pop_size
                    player.set_reward(player.get_reward()/self.pop_size)
                if player.get_strategyType() == "Tit-for-Tat":
                    if self.previous_average is None:
                        target -= player.get_reward() / self.pop_size
                        player.set_reward(player.get_reward() / self.pop_size)
                    else:
                        target -= self.previous_average
                        player.set_reward(self.previous_average)
                if player.get_strategyType() == "Revenge":
                    if not player.get_hurt():
                        target -= player.get_reward()/self.pop_size
                        player.set_reward(player.get_reward() / self.pop_size)
            probability_of_punishment = self.probability_of_punishment
            punishment = 1-min(max(self.punishment_amount, 0), 1)
            if target > 0:
                probability_of_punishment += self.risk_function(self.risk_function(target/self.target))
                probability_of_punishment = min(max(probability_of_punishment, 0), 1)
                self.probability_of_punishment += 0.05
                self.punishment_amount += 0.05
                for player in group:
                    player.set_hurt(True)
                    result = np.random.choice([True, False], 1,
                                              p=[probability_of_punishment, 1 - probability_of_punishment])
                    if result and 0 <= punishment <= 1:
                        player.set_reward((player.get_reward() + player.get_income()) * punishment)
                    else:
                        player.set_reward(player.get_reward() + player.get_income())
            else:
                self.probability_of_punishment -= 0.05
                self.punishment_amount -= 0.05
                for player in group:
                    player.set_reward(player.get_reward() + player.get_income())
            self.previous_average = target/self.pop_size
            # if player.get_action() == 'C' :
            #     player.set_reward(payC)  # the reward in the current round
            #     player.set_fitness(payC) # the accumulted rewards
            # else:
            #     player.set_reward(payD)
            #     player.set_fitness(payD)

    # Randomly ditribute players into different groups and play game
    def grouping_playing(self):
        population = self.population
        group_size = self.group_size
        pop_size = self.pop_size
        shuffle(population)
    
        i = 0
        while i < pop_size:
            if i + group_size < pop_size:
                group = population[i:i+group_size]
            else:
                group = population[i:pop_size]
            self.play_game(group)
            i = i + group_size

    def random_list(self):
        pop_update_rate = self.pop_update_rate
        pop_size = self.pop_size
        update_lst = []
        if pop_update_rate == pop_size:
            update_lst = np.random.permutation(pop_size)
        else:
            while len(update_lst) != pop_update_rate:
                i = random.randint(0,pop_size-1)
                if i not in update_lst:
                    update_lst.append(i)
        return update_lst

    def inverse_action(self,action):
        act = None
        if action == 'C':
            act = 'D'
        else:
            act = 'C'
        return act
        
    def update_population(self, groups):
        for group in groups:
            top_1 = -1
            top_2 = -1
            bottom_1 = math.inf
            bottom_2 = math.inf
            strategy_1 = None
            strategy_2 = None
            for player in group:
                if player.get_reward() > top_1:
                    top_2 = top_1
                    strategy_2 = strategy_1
                    top_1 = player.get_reward()
                    strategy_1 = player.get_strategyType()
                elif player.get_reward() > top_2:
                    top_2 = player.get_reward()
                    strategy_2 = player.get_strategyType()
                if player.get_reward() < bottom_1:
                    bottom_2 = bottom_1
                    bottom_1 = player.get_reward()
                elif player.get_reward() > bottom_2:
                    bottom_2 = player.get_reward()
            x = 0
            while x < 2:
                for player in group:
                    if player.get_reward() == bottom_2:
                        bottom_2 = None
                        player.set_strategyType(strategy_2)
                        break
                    if player.get_reward() == bottom_1:
                        bottom_1 = None
                        player.set_strategyType(strategy_1)
                        break
                x += 1


        # pop_size = self.pop_size
        # pop_update_rule = self.pop_update_rule
        # population = self.population
        # cooperator_count = self.cooperator_count
        # defector_count = self.defector_count
        # cooperators = self.cooperators
        # defectors = self.defectors
        # mutation_rate = self.mutation_rate
        # cooperator_proportion = cooperator_count / pop_size
        # update_lst = self.random_list()
        # # the players in update_lst needs to be updated.
        # for i in update_lst:
        #     # update play i with play j using different update rules
        #     player_i = population[i]
        #     action_old = player_i.get_action()
        #     if pop_update_rule == 'random':
        #         j = i
        #         while j == i:
        #             j = random.randint(0,pop_size-1)
        #         player_j = population[j]
        #         action_new = player_j.get_action()
        #
        #
        #     elif pop_update_rule == 'popular':
        #         if cooperator_proportion >= 0.5:
        #             action_new = 'C'
        #         else:
        #             action_new = 'D'
        #
        #     elif pop_update_rule == 'accumulated_best':
        #         player_j = player_i
        #         fitness = player_i.get_fitness()
        #         for player in population:
        #             if player.get_fitness() > fitness:
        #                 player_j = player
        #                 fitness = player.get_fitness()
        #         action_new = player_j.get_action()
        #
        #     elif pop_update_rule == 'accumulated_worst':
        #         player_j = player_i
        #         fitness = player_i.get_fitness()
        #         for player in population:
        #             if player.get_fitness() < fitness:
        #                 player_j = player
        #                 fitness = player.get_fitness()
        #         action_new = player_j.get_action()
        #
        #     elif pop_update_rule == 'accumulated_better':
        #         lst = []
        #         fitness = player_i.get_fitness()
        #         for player in population:
        #             if player.get_fitness() >= fitness:
        #                 lst.append(player)
        #         player_j = random.choice(lst)
        #         action_new = player_j.get_action()
        #
        #     elif pop_update_rule == 'current_best':
        #         player_j = player_i
        #         reward = player_i.get_reward()
        #         for player in population:
        #             if player.get_reward() > reward:
        #                 player_j = player
        #                 reward = player.get_reward()
        #         action_new = player_j.get_action()
        #
        #     elif pop_update_rule == 'current_worst':
        #         player_j = player_i
        #         reward = player_i.get_reward()
        #         for player in population:
        #             if player.get_reward() < reward:
        #                 player_j = player
        #                 reward = player.get_reward()
        #         action_new = player_j.get_action()
        #
        #     elif pop_update_rule == 'current_better':
        #         lst = []
        #         reward = player_i.get_reward()
        #         for player in population:
        #             if player.get_reward() >= reward:
        #                 lst.append(player)
        #         player_j = random.choice(lst)
        #         action_new = player_j.get_action()
        #
        #     elif pop_update_rule == 'fermi':
        #         j = i
        #         while j == i:
        #             j = random.randint(0,pop_size-1)
        #         player_j = population[j]
        #         beta = 1    # beta --> 0 random/neutral  beta >> 1  reduces to a step function
        #         fitDelta = player_j.get_fitness() - player_i.get_fitness()
        #         prob = 1.0 / (1.0 + math.exp( -1 * beta * fitDelta))
        #         if prob > random.random():
        #             action_new = player_j.get_action()
        #         else:
        #             action_new = player_i.get_action()
        #     else:
        #         assert(False)
        #
        #     # mutation
        #     if random.random() < mutation_rate:
        #         if action_new == 'C':
        #             action_new = 'D'
        #         elif action_new == 'D':
        #             action_new = 'C'
        #         else:
        #             assert(False)
        #     # set the new action to player i
        #     player_i.set_action(action_new)
        #
        #
        #     if action_new != action_old:
        #         if action_old == 'D':
        #             cooperator_count = cooperator_count + 1
        #             defector_count = defector_count - 1
        #         elif action_old == 'C':
        #             cooperator_count = cooperator_count - 1
        #             defector_count = defector_count + 1
        #         else:
        #             assert(False)
        #
        #     #population[i] = player_i
        #
        # cooperators.append(cooperator_count / pop_size)
        # defectors.append(defector_count / pop_size)
        #
        # self.cooperators = cooperators
        # self.defectors = defectors
        # self.population = population
        # self.cooperator_count = cooperator_count
        # self.defector_count = defector_count


def fileSaving(filename, data, writing_model):
    #i = 0
    with open(filename, writing_model) as f:
        f_csv=csv.writer(f, quotechar= '|', quoting = csv.QUOTE_MINIMAL)
        for line in data:
            f_csv.writerow(line)

# main function

if __name__ == '__main__':

    #pop_update_rule_list = ['random', 'popular', 'fpopular', 'obstinate', 'mix', 'qlearning', 'rdexample', 'fbexample', 'obexample', 'rewardModel']
    pop_update_rule_list = ['random', 'popular', 'accumulated_best', 'accumulated_worst', 'accumulated_better', 'current_best', 'current_worst', 'current_better', 'fermi']
    #pop_update_rule_list = ['random', 'popular', 'accumulated_best']
    
    pop_update_rate_list = [1]          # the number of player that will copy other strategies in each round
    
    pop_size_list = [500, 1000, 1500, 2000]          # population size
    #pop_size_list = [100]          # population size
    
    cost_list = [0.01]               # cost
    
    F_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # enhancement factor: value * group_size
    #F_list = [0.2] # enhancement factor
    
    group_size_list = [0.01, 0.05, 0.2, 1]         # the size of group_size  value * pop_size
    #group_size_list = [0.2]         # the size of group
    
    M_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]      # the threshold for participants: minimum number of cooperators to play the game: value * group_size
    #M_list = [0.2]
    
    mutation_rate_list = [0.001]  # the probability of mutation
    
    runs = 50                    # total number of independent runs
    #runs = 1

    tmax = 100000   #10^5
    #tmax = 1000
    
    time_list = range(0,100100,1000) # 1e05, 100 time points
    #time_list = range(0,1001,10) # 1e03, 100 time points

    # write all data every 10th time_list step
    # (and not at first and last, already done specifically)
    #snapshot_time_list = [time_list[i] for i in range(1, len(time_list)-1, len(time_list)/10)]

    games = game(10, None, None, None, None, None, None, 20, "Linear")
    groups = games.init_population()
    string = ""
    for player in games.get_population():
        string = string + str(player)
    string = string + "\n"
    f = open("test.txt", "w")
    f.write(string)
    f.close()
    for x in range(1000):
        games.play_game(groups)
        games.update_population(groups)
        string = ""
        for player in games.get_population():
            string = string + str(player)
        string = string + "\n"
        f = open("test.txt", "a")
        f.write(string)
        f.close()



    # param_list = [] # list of parameter tuples: build list then run each
    # param_count = 0
    # for pop_size in pop_size_list:
    #     for pop_update_rule in pop_update_rule_list:
    #         for pop_update_rate in pop_update_rate_list:
    #             for group_size in group_size_list:
    #                 group_size = int (round(group_size * pop_size))
    #                 for F in F_list:
    #                     F = int (round(F * group_size))
    #                     for M in M_list:
    #                         M = int (M * round(group_size))
    #                         for cost in cost_list:
    #                             cost = cost / (F+1)
    #                             for mutation_rate in mutation_rate_list:
    #                                 for run_number in range(runs):
    #                                     param_count += 1
    #                                     param_list.append((pop_size, pop_update_rule, pop_update_rate, F, cost, M, group_size, mutation_rate, run_number))
    #
    # print (len(param_list),'of total',param_count,'models to run')
    # print (int(math.ceil(float(len(param_list))/1)),' models per MPI task')
    # #print 'time series: writing total',len(time_list)*param_count,'time step records'
    #
    #
    # # Write out current configuration
    # try:
    #     os.mkdir('./results/')
    # except OSError as e:
    #     if e.errno == errno.EEXIST:
    #         pass
    #
    # # now that we have list of parameter tuples, execute in parallel
    # # using MPI: each MPI process will process
    # # ceil(num_jobs / mpi_processes) of the parameter tuples in the list
    # num_jobs = len(param_list)
    # job_i = rank
    #
    # txt = 'results/' + 'result'+ str(rank) + '.csv'
    # fileSaving(txt, [], 'w')
    #
    # while job_i < num_jobs:
    #     (pop_size, pop_update_rule, pop_update_rate, F, cost, M, group_size, mutation_rate, run_number) = param_list[job_i]
    #
    #     # In this format, the tuple after 'rank 0: '
    #     # can be cut&pasted into resumeFrom: command line
    #     #sys.stdout.write('rank %d: %d,%s,%d,%d,%f,%d,%d,%d\n' % (rank, pop_size, pop_update_rule, pop_update_rate, F, cost, M, group_size, run_number)
    #
    #     random.seed()
    #     rounds = []
    #     #txt = 'NPlayGame' + '_' + 'pop_size' + '_' + str(pop_size)+'_'+ pop_update_rule
    #
    #     games = game(pop_size, pop_update_rule, pop_update_rate, F, cost, M, group_size, mutation_rate)
    #     games.init_population()
    #     population = games.get_population()
    #
    #     rounds.append(0)
    #     games.csvDataBuilder(run_number, rounds[0])
    #
    #     for roundx in range(1,tmax) :
    #         games.grouping_playing()
    #         rounds.append(roundx)
    #         games.update_population(roundx)
    #         if roundx in time_list:
    #             cooperators = games.get_cooperators()
    #             games.average_reward()
    #             games.csvDataBuilder(run_number, roundx)
    #     cooperators = games.get_cooperators()
    #     games.csvDataBuilder(run_number, roundx)
    #
    #     txt = 'results/' + 'result'+ str(rank) + '.csv'
    #     fileSaving(txt, games.get_lst(), 'a')
    #     #stat_plots(rounds, cooperators, txt)
    #
    #     job_i += mpi_processes



