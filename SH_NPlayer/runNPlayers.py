# coding=UTF-8

from __future__ import division
from Player import Player
from random import shuffle
import random
import matplotlib.pyplot as plt
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
        self.failures = 0
        self.success = 0

    def set_failures(self, fails):
        self.failures = fails

    def set_success(self, success):
        self.success = success

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
                player = Player(pop_size, None, reward=4.0, fitness=0.0, income = 4.0, rich = True, strategyType= np.random.choice(strategies))
                rich_group.append(player)
            else :                                
                player = Player(pop_size, None, reward=2.0, fitness=0.0, income = 2.0, rich = False, strategyType= np.random.choice(strategies))
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
                contribution = player.get_reward()
                if player.get_strategyType() == "Nothing":
                    player.set_previous_Contribution(0)
                elif player.get_strategyType() == "Everything":
                    target -= contribution
                    player.set_previous_Contribution(contribution)
                    player.set_reward(0)
                elif player.get_strategyType() == "FairShare":
                    target -= contribution//self.pop_size
                    player.set_previous_Contribution(contribution//self.pop_size)
                    player.set_reward(player.get_reward() - (contribution//self.pop_size))
                elif player.get_strategyType() == "Tit-for-Tat":
                    if self.previous_average is None:
                        target -= contribution // self.pop_size
                        player.set_previous_Contribution(player.get_reward() // self.pop_size)
                        player.set_reward(player.get_reward() - (contribution//self.pop_size))
                    else:
                        target -= min(self.previous_average, player.get_reward())
                        player.set_previous_Contribution(min(self.previous_average, player.get_reward()))
                        player.set_reward(player.get_reward() - min(self.previous_average, player.get_reward()))
                elif player.get_strategyType() == "Revenge":
                    if not player.get_hurt():
                        target -= contribution//self.pop_size
                        player.set_reward(player.get_reward() - (contribution//self.pop_size))
                    else:
                        player.set_previous_Contribution(0)
                else:
                    print("this is not a strategy")
                    print(type(player.get_strategyType()))
                    print(player.get_strategyType()[0])
                    print(player.get_strategyType())

                    sys.exit(1)
            probability_of_punishment = self.probability_of_punishment
            punishment = 1-min(max(self.punishment_amount, 0), 1)
            if target > 0:
                self.failures += 1
                probability_of_punishment += self.risk_function(self.risk_function(target/self.target))
                probability_of_punishment = min(max(probability_of_punishment, 0), 1)
                self.probability_of_punishment += 0.05
                self.punishment_amount += 0.05
                for player in group:
                    player.set_hurt(True)
                    result = np.random.choice([True, False], 1,
                                              p=[probability_of_punishment, 1 - probability_of_punishment])
                    if result and 0 <= punishment < 1 and player.get_moral_factor(self.target) < 1:
                        player.set_reward((player.get_reward() + player.get_income()) * max(punishment-player.get_moral_factor(self.target), 0))
                    elif result and 0 <= punishment < 1:
                        player.set_reward((player.get_reward() + player.get_income()) * punishment)
                    else:
                        if player.get_moral_factor(self.target) < 1:
                            player.set_reward((player.get_reward() + player.get_income())*player.get_moral_factor(self.target))
                        else:
                            player.set_reward(player.get_reward() + player.get_income())
            else:
                self.success += 1
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
        all_strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
        for group in groups:
            top_1 = None
            top_2 = None
            bottom_1 = None
            bottom_2 = None
            strategy_1 = None
            strategy_2 = None
            for player in group:
                if top_1 is None or player.get_reward() > top_1:
                    top_2 = top_1
                    strategy_2 = np.array([strategy_1]).tolist()
                    top_1 = player.get_reward()
                    strategy_1 = np.array([player.get_strategyType()]).tolist()
                elif top_2 is None or player.get_reward() > top_2:
                    top_2 = player.get_reward()
                    strategy_2 = np.array([player.get_strategyType()]).tolist()
                if bottom_1 is None or player.get_reward() < bottom_1:
                    bottom_2 = bottom_1
                    bottom_1 = player.get_reward()
                elif bottom_2 is None or player.get_reward() > bottom_2:
                    bottom_2 = player.get_reward()
            x = 0
            while x < 2:
                for player in group:
                    if player.get_reward() == bottom_2:
                        bottom_2 = None
                        random_strategy = [random.choice(all_strategies)]
                        player.set_strategyType(np.random.choice([strategy_2[0], random_strategy[0]],
                                              p=[1-self.M, self.M]))
                        break
                    if player.get_reward() == bottom_1:
                        bottom_1 = None
                        random_strategy = [random.choice(all_strategies)]
                        player.set_strategyType(np.random.choice([strategy_1[0], random_strategy[0]],
                                              p=[1-self.M, self.M]))
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

    games = game(5, None, None, None, None, 0.03, None, 20, "Linear")
    groups = games.init_population()
    string = ""
    for player in games.get_population():
        string = string + str(player)
    string = string + "\n"
    f = open("test.txt", "w")
    f.write(string)
    f.close()
    wealth = []
    money = []
    proportion = []
    strategies = []
    for player in games.get_population():
        string = string + str(player)
        money.append(player.get_reward())
        strategies.append(player.get_strategyType())
    proportion.append(strategies)
    wealth.append(money)
    successes = []
    for x in range(10):
        games.play_game(groups)
        games.update_population(groups)
        string = ""
        money = []
        strategies = []
        for player in games.get_population():
            string = string + str(player)
            money.append(player.get_reward())
            strategies.append(player.get_strategyType())
        successes.append([games.failures, games.success])
        games.set_success(0)
        games.set_failures(0)
        proportion.append(strategies)
        wealth.append(money)
        string = string + "\n"
        f = open("test.txt", "a")
        f.write(string)
        f.close()
    for y in range(5):
        line = []
        for x in range(10):
            line.append(wealth[x][y])
        plt.plot(range(10), line, label = "player" + str(y))
    plt.legend()
    plt.xlabel('Number of rounds')
    plt.ylabel('Wealth')
    plt.savefig('wealth.png')
    plt.figure()
    all_strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
    for strat in all_strategies:
        line = []
        for y in range(10):
            line.append(proportion[y].count(strat))
        plt.plot(range(10), line, label= strat)
    plt.legend()
    plt.xlabel('Number of rounds')
    plt.ylabel('Number of players using the strategy')
    plt.savefig('strategy.png')
    plt.figure()
    print(successes)
    print([x[0] for x in successes])
    plt.bar(np.arange(10), [x[0] for x in successes], color = 'r', width = 0.25 )
    plt.bar(np.arange(10)+0.25,[y[1] for y in successes], color = 'g', width = 0.25)
    plt.savefig('games.png')


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



