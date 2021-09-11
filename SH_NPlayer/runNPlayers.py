# coding=UTF-8

from __future__ import division
from Player import Player
from learner import NashQLearner
from random import shuffle
import random
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import sys
import math
from tqdm import tqdm
import time
import csv
import os,errno
import os.path
import shutil
import seaborn as sns
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

class game(object):
    def __init__(self, pop_size, rounds, strategies, number_of_rich, probability_of_punishment, punishment_amount, target, riskType, mutations):
        self.pop_size = pop_size
        self.target = target
        self.population = []
        self.M = mutations
        self.riskType = riskType
        self.probability_of_punishment = probability_of_punishment
        self.punishment_amount = punishment_amount
        self.previous_average = 0
        self.rounds = rounds
        self.failures = 0
        self.success = 0
        self.rich_proportions = number_of_rich
        assert(sum(self.rich_proportions)==1 and len(self.rich_proportions)==2)
        self.strategy_proportions = strategies
        assert (sum(self.strategy_proportions) == 1  and len(self.strategy_proportions)==6)
        self.number_of_learners = 0
        self.learners = []

    def set_failures(self, fails):
        self.failures = fails

    def set_success(self, success):
        self.success = success

    def get_population(self):
        return self.population

    # calculate the average reward of each round.
    def average_reward(self):
        population = self.population
        pop_size = self.pop_size
        totalreward = 0
        for player in population:
            totalreward = totalreward + player.get_reward()
        self.avReward = totalreward/pop_size

    def init_population(self, morals):
        population = []
        strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
        rich_group = []
        poor_group = []
        pop_size = self.pop_size - self.number_of_learners
        poor_players = math.ceil(self.pop_size*self.rich_proportions[0])
        rich_players = math.floor(self.pop_size*self.rich_proportions[1])
        rich_strategies = [round(x*rich_players) for x in self.strategy_proportions]
        poor_strategies = [round(y*poor_players) for y in self.strategy_proportions]
        self.pop_size = sum(rich_strategies) + sum(poor_strategies)
        index = 0
        id = 1
        for strat in rich_strategies:
            for x in range(strat):
                if index == 5:
                    player = NashQLearner(id = id,  reward=4.0, income = 4.0, moral = morals, rich = True)
                    self.learners.append(player)
                else:
                    player = Player(pop_size, None, reward=4.0, id = id, income = 4.0, moral = morals, rich = True, strategyType= strategies[index])
                    rich_group.append(player)
                id += 1
                population.append(player)
            index += 1
        index = 0
        for strat in poor_strategies:
            for x in range(strat):
                if index == 5:
                    player = NashQLearner(id = id, reward=2.0, income=2.0, moral = morals, rich=False)
                    self.learners.append(player)
                else:
                    player = Player(pop_size, None, reward=2.0, id = id, income = 2.0, moral = morals, rich = False,  strategyType= strategies[index])
                    poor_group.append(player)
                id += 1
                population.append(player)
            index += 1
        self.population = population
        groups = [rich_group, poor_group, self.learners]
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
        group = groups[0]+groups[1]+groups[2]
        learners = groups[2]
        target = self.target
        for x in range(self.rounds):
            contribution = 0
            players = 0
            for player in group:
                players += 1
                player.set_previous_reward(player.get_reward())
                if player.get_strategyType() == "Nothing":
                    if np.random.choice([True, False], 1, p=[player.get_morals(), 1 - player.get_morals()]):
                        contribution, target = self.moral_contribution(contribution, player, target)
                    else:
                        player.set_previous_Contribution(0)
                elif player.get_strategyType() == "Everything":
                    if np.random.choice([True, False], 1, p=[player.get_morals(), 1 - player.get_morals()]):
                        contribution, target = self.moral_contribution(contribution, player, target)
                    else:
                        target -= contribution
                        player.set_previous_Contribution(contribution)
                        player.set_reward(0)
                        contribution += 1
                elif player.get_strategyType() == "FairShare":
                    if np.random.choice([True, False], 1, p=[player.get_morals(), 1 - player.get_morals()]):
                        contribution, target = self.moral_contribution(contribution, player, target)
                    else:
                        target -= player.get_reward()//self.pop_size
                        player.set_previous_Contribution(player.get_reward()//self.pop_size)
                        if player.get_reward() != 0:
                            contribution += (player.get_reward() // self.pop_size) / player.get_reward()
                        player.set_reward(player.get_reward() - (player.get_reward()//self.pop_size))
                elif player.get_strategyType() == "Tit-for-Tat":
                    if self.previous_average is None:
                        target -= player.get_reward() // self.pop_size
                        player.set_previous_Contribution(player.get_reward() // self.pop_size)
                        if player.get_reward() != 0:
                            contribution += (player.get_reward() // self.pop_size) / player.get_reward()
                        player.set_reward(player.get_reward() - (player.get_reward() // self.pop_size))
                    elif np.random.choice([True, False], 1, p=[player.get_morals(), 1 - player.get_morals()]):
                        contribution, target = self.moral_contribution(contribution, player, target)
                    else:
                        target -= min(self.previous_average, player.get_reward())
                        player.set_previous_Contribution(min(self.previous_average, player.get_reward()))
                        if player.get_reward() != 0:
                            contribution += (min(self.previous_average, player.get_reward())) / player.get_reward()
                        player.set_reward(player.get_reward() - min(self.previous_average, player.get_reward()))
                elif player.get_strategyType() == "Revenge":
                    if not player.get_hurt():
                        if np.random.choice([True, False], 1, p=[player.get_morals(), 1 - player.get_morals()]):
                            contribution, target = self.moral_contribution(contribution, player, target)
                        else:
                            target -= player.get_reward() // self.pop_size
                            player.set_previous_Contribution(player.get_reward() // self.pop_size)
                            if player.get_reward() != 0:
                                contribution += (player.get_reward() // self.pop_size) / player.get_reward()
                            player.set_reward(player.get_reward() - (player.get_reward() // self.pop_size))
                    else:
                        player.set_previous_Contribution(0)
                elif player.get_strategyType() == "Learner":
                    action = player.act()
                    amount_paid = min(((action/4)*player.get_reward()), self.target)

                    player.set_previous_Contribution(amount_paid)
                    player.set_reward(player.get_reward() - amount_paid)
                else:
                    print("this is not a strategy")
                    print(type(player.get_strategyType()))
                    print(player.get_strategyType()[0])
                    print(player.get_strategyType())

                    sys.exit(1)
            probability_of_punishment = self.probability_of_punishment
            punishment = 1-min(max(self.punishment_amount, 0), 1)
            punish = 0
            if target > 0:
                self.failures += 1
                probability_of_punishment += self.risk_function(self.risk_function(target / self.target)).real
                probability_of_punishment = min(max(probability_of_punishment, 0), 1)
                self.probability_of_punishment += 0.05
                self.punishment_amount += 0.05
                for player in group:
                    player.set_hurt(True)
                    result = np.random.choice([True, False], 1,
                                              p=[probability_of_punishment, 1 - probability_of_punishment])
                    loss = player.get_previous_reward()
                    if result and 0 <= punishment < 1 and player.get_moral_factor(self.target/self.pop_size) < 1:
                        player.set_reward((player.get_reward() + player.get_income()) * max(punishment-player.get_moral_factor(self.target/self.pop_size), 0))
                    elif result and 0 <= punishment < 1:
                        player.set_reward((player.get_reward() + player.get_income()) * punishment)
                    else:
                        if player.get_moral_factor(self.target/self.pop_size) < 1:
                            player.set_reward((player.get_reward() + player.get_income())*(1-player.get_moral_factor(self.target)))
                        else:
                            player.set_reward(player.get_reward() + player.get_income())
                    if player.get_strategyType() != "Learner":
                        punish -= loss - player.get_reward()
            else:
                self.success += 1
                self.probability_of_punishment -= 0.05
                self.punishment_amount -= 0.05
                for player in group:
                    player.set_reward(player.get_reward() + player.get_income())
            for player in learners:
                c = contribution
                p = players
                pun = punish
                for others in learners:
                    if others.get_id() != player.get_id():
                        p += 1
                        if others.get_previous_reward() != 0:
                            loss = player.get_previous_reward()
                            c += others.get_previous_Contribution()/others.get_previous_reward()
                            pun -= loss - player.get_reward()
                o_action = find_action(c / p)
                player.observe(reward=round(player.get_previous_reward()-player.get_reward()), reward_o=round(pun), opponent_action=o_action)
                if player == learners[0]:
                    q_1, q_2 = [], []
                    for action1 in player.actions:
                        row_q_1, row_q_2 = [], []
                        for action2 in player.actions:
                            joint_action = (action1, action2)
                            row_q_1.append(player.q["nonstate"][joint_action])
                            row_q_2.append(player.q_o["nonstate"][joint_action])
                        q_1.append(row_q_1)
                        q_2.append(row_q_2)
                    plt.matshow([q_1, q_2])
            self.previous_average = target/self.pop_size

    def moral_contribution(self, contribution, player, target):
        amount_paid = min(player.get_morals() * player.get_reward(), max(target, 0))
        target -= amount_paid
        if amount_paid != 0:
            contribution += amount_paid / player.get_reward()
        player.set_previous_Contribution(amount_paid)
        player.set_reward(player.get_reward() - amount_paid)
        return contribution, target

    def update_population(self, groups):
        all_strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
        pop = [groups[0], groups[1]]
        for group in pop:
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

def find_action(x):
    if x <= 0.125:
        return 0
    elif x <= 0.375:
        return 1
    elif x <= 0.625:
        return 2
    elif x <= 0.875:
        return 3
    else:
        return 4

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
    pun_0 = []
    pun_1 = []
    pun_2 = []
    pun_3 = []
    pun_4 = []
    pun_5 = []
    pun_6 = []
    pun_7 = []
    pun_8 = []
    pun_9 = []
    pun_10 = []
    morals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    if rank != 0:
        morals = morals[rank-1]
        for punishment in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            target = 280
            games = game(100, 10, [0.12, 0.12, 0.22, 0.22, 0.22, 0.1], [0.8, 0.2], 0.5, 0.2, target, "Linear", 0.03)
            groups = games.init_population(morals)
            string = "Morals-"+str(morals)+"-punishment_amount-"+str(punishment)+"-hard_target-"
            # for player in games.get_population():
            #    string = string + str(player)
            # string = string + "\n"
            # f = open("test.txt", "w")
            # f.write(string)
            # f.close()
            wealth = []
            money = []
            proportion = []
            strategies = []
            for player in games.get_population():
                money.append(player.get_reward())
                strategies.append(player.get_strategyType())
            proportion.append(strategies)
            wealth.append(money)
            successes = []
            number_of_games = 100
            for x in range(number_of_games):
                games.play_game(groups)
                games.update_population(groups)
                money = []
                strategies = []
                for player in games.get_population():
                    money.append(player.get_reward())
                    strategies.append(player.get_strategyType())
                successes.append([games.failures, games.success])
                games.set_success(0)
                games.set_failures(0)
                proportion.append(strategies)
                wealth.append(money)
                #string = string + "\n"
                #f = open("test.txt", "a")
                #f.write(string)
                #f.close()

            agent_wealth = []
            for x in range(number_of_games):
                agent_wealth.append(sum(wealth[x])/len(games.get_population()))
            plt.plot(range(number_of_games), agent_wealth, label = "average wealth of the players")
            plt.legend()
            plt.xlabel('Number of games')
            plt.ylabel('Wealth')
            name = string + 'wealth.png'
            plt.savefig(name)
            plt.figure()
            all_strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
            for strat in all_strategies:
                line = []
                for y in range(number_of_games):
                    line.append(proportion[y].count(strat)/games.pop_size)
                plt.plot(range(number_of_games), line, label= strat)
            plt.legend()
            name = string + 'strategy.png'
            plt.xlabel('Number of games')
            plt.ylabel('Proportion of players using the strategy')
            plt.savefig(name)
            plt.figure(figsize=(9, 5))
            if punishment == 0.0:
                pun_0.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.1:
                pun_1.append(sum(agent_wealth) / number_of_games)
            if punishment == 0.2:
                pun_2.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.3:
                pun_3.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.4:
                pun_4.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.5:
                pun_5.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.6:
                pun_6.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.7:
                pun_7.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.8:
                pun_8.append(sum(agent_wealth)/number_of_games)
            if punishment == 0.9:
                pun_9.append(sum(agent_wealth)/number_of_games)
            if punishment == 1:
                pun_10.append(sum(agent_wealth)/number_of_games)
            plt.bar(np.arange(number_of_games), [x[0] for x in successes], color = 'r', width = 0.5, align='edge' )
            plt.bar(np.arange(number_of_games)+0.5, [y[1] for y in successes], color = 'g', width = 0.5, align='edge')
            name = string + 'games.png'
            plt.xlabel('Number of games')
            plt.ylabel('Number of rounds that failed/succeed')
            plt.savefig(name)
            plt.close('all')
        heat = [pun_0, pun_1, pun_2, pun_3, pun_4, pun_5, pun_6, pun_7,pun_8, pun_9, pun_10]
        comm.send(heat, dest=0)
    else:
        for worker in range(1, size):
            data = comm.recv(source=worker)
            if worker == 1:
                results = data
            else:
                i = 0
                while i < len(results):
                    results[i] = results[i] + data[i]
                    i += 1
        ax = sns.heatmap(results, xticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],yticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xlabel('Moral factor')
        ax.set_ylabel('Punishment amount')
        figure = ax.get_figure()
        figure.savefig('heatmap-hard_target".png')


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



