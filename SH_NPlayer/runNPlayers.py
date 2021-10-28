# coding=UTF-8
# ------------------
# Attribution Information: This code was developed off work done by Michael Kirley

from __future__ import division
from Player import Player
from learner import NashQLearner
import random
import matplotlib.pyplot as plt
import sys
import math
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
        # assert (sum(self.strategy_proportions) == 1 and len(self.strategy_proportions) == 6)
        self.number_of_learners = 0
        self.learners = []
        self.round = 1

    def set_failures(self, fails):
        self.failures = fails

    def set_success(self, success):
        self.success = success

    def set_punishment(self, punish):
        self.punishment_amount = punish

    def set_punishment_prob(self, punish_prob):
        self.probability_of_punishment = punish_prob

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

    # Initialise the players
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
        # Initialise the rich players first and then the poor ones
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

    # Choose which risk_function to use
    def risk_function(self, x):
        if self.riskType == "Linear":
            return 1-x
        if self.riskType == "Low Probability Power function":
            return 1-(x**0.1)
        if self.riskType == "High Probability Power function":
            return 1-(x**10)
        if self.riskType == "Curves exhibiting threshold effects":
            return 1 / (math.e ** (10 * (x - 1 / 2)) + 1)

    # PLay a game
    def play_game(self, groups):
        group = groups[0]+groups[1]+groups[2]
        learners = groups[2]
        target = self.target
        # Play R rounds
        for x in range(self.rounds):
            contribution = 0
            players = 0
            for player in group:
                players += 1
                player.set_previous_reward(player.get_reward())
                # Get the agent contributions based on their strategies
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
            # Calculate if the agents reached the target goal or not
            if target > 0:
                # If not punish the agents
                self.failures += 1
                probability_of_punishment += self.risk_function(self.risk_function(target / self.target)).real
                probability_of_punishment = min(max(probability_of_punishment, 0), 1)
                self.probability_of_punishment += 0.005
                self.punishment_amount += 0.001
                for player in group:
                    player.set_hurt(True)
                    result = np.random.choice([True, False], 1,
                                              p=[probability_of_punishment, 1 - probability_of_punishment])
                    loss = player.get_previous_reward()
                    if result and 0 <= punishment < 1 and player.get_moral_factor(self.target/self.pop_size) < 1:
                        if player.get_rich():
                            player.set_reward((player.get_reward() + player.get_income()) * max((punishment/2)-player.get_moral_factor(self.target/self.pop_size), 0))
                        else:
                            player.set_reward((player.get_reward() + player.get_income()) * max(
                                punishment - player.get_moral_factor(self.target / self.pop_size), 0))
                    elif result and 0 <= punishment < 1:
                        if player.get_rich():
                            player.set_reward((player.get_reward() + player.get_income()) * (punishment/2))
                        else:
                            player.set_reward((player.get_reward() + player.get_income()) * punishment)
                    else:
                        if player.get_moral_factor(self.target/self.pop_size) < 1:
                            player.set_reward((player.get_reward() + player.get_income())*(1-player.get_moral_factor(self.target)))
                        else:
                            player.set_reward(player.get_reward() + player.get_income())
                    if player.get_strategyType() != "Learner":
                        punish -= loss - player.get_reward()
            else:
                # Else no punishment
                self.success += 1
                self.probability_of_punishment -= 0.005
                self.punishment_amount -= 0.001
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
                # This print the q table that the reinforcement learners are learning
                # if player == learners[0] and self.round%20 == 1:
                #     q_1, q_2 = [], []
                #     for action1 in player.actions:
                #         row_q_1, row_q_2 = [], []
                #         for action2 in player.actions:
                #             joint_action = (action1, action2)
                #             row_q_1.append(player.q["nonstate"][joint_action])
                #             row_q_2.append(player.q_o["nonstate"][joint_action])
                #         q_1.append(row_q_1)
                #         q_2.append(row_q_2)
                #     plt.matshow(q_1, fignum=False)
                #     plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 0.25, 0.5, 0.75, 1])
                #     plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 0.25, 0.5, 0.75, 1])
                #     plt.xlabel('Player actions (Percent of wealth contributed)')
                #     plt.ylabel('Opponent actions (Percent of wealth contributed)')
                #     plt.title("Player payoff")
                #     name = "Player_Payoff_matrix_after_round_" + str(self.round)
                #     plt.savefig(name)
                #     plt.matshow(q_2, fignum=False)
                #     plt.xticks(ticks = [0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 0.25, 0.5, 0.75, 1])
                #     plt.yticks(ticks = [0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 0.25, 0.5, 0.75, 1])
                #     plt.xlabel('Player actions (Percent of wealth contributed)')
                #     plt.ylabel('Opponent actions (Percent of wealth contributed)')
                #     plt.title("Opponent payoff")
                #     name = "Opponent_Payoff_matrix_after_round_" + str(self.round)
                #     plt.savefig(name)
            self.round += 1
            self.previous_average = target/self.pop_size
            target = self.target
    # Get the agent to make a contribute based off their fairness factor
    def moral_contribution(self, contribution, player, target):
        amount_paid = min(player.get_morals() * player.get_reward(), max(target, 0))
        target -= amount_paid
        if amount_paid != 0:
            contribution += amount_paid / player.get_reward()
        player.set_previous_Contribution(amount_paid)
        player.set_reward(player.get_reward() - amount_paid)
        return contribution, target
    # Give the agents a chance to evolve their strategies
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

# main function

if __name__ == '__main__':
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
    coop_0 = []
    coop_1 = []
    coop_2 = []
    coop_3 = []
    coop_4 = []
    coop_5 = []
    coop_6 = []
    coop_7 = []
    coop_8 = []
    coop_9 = []
    coop_10 = []
    morals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    morals = morals[rank%11]
    for punishment in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        number_of_games = 2
        string = "Morals-" + str(morals) + "-punishment_amount-" + str(punishment) + "-hard_target-"
        successes = [0, 0]
        for i in range(10):
            target = 100
            games = game(100, 10, [0.12, 0.12, 0.22, 0.22, 0.22, 0.1], [0.8, 0.2], 0.5, punishment, target, "Linear", 0.03)
            groups = games.init_population(morals)
            money = []
            strategies = []
            for player in games.get_population():
                money.append(player.get_reward())
                strategies.append(player.get_strategyType())
            if i == 0:
                wealth = [money]
                proportion = [[strategies]]
                wins = []
                losses = []
            else:
                wealth[0] = wealth[0] + money
                proportion[0] = proportion[0] + [strategies]
            for x in range(number_of_games):
                games.play_game(groups)
                games.update_population(groups)
                money = []
                strategies = []
                for player in games.get_population():
                    money.append(player.get_reward())
                    strategies.append(player.get_strategyType())
                if i == 0:
                    proportion.append([strategies])
                    wealth.append(money)
                else:
                    wealth[x+1] = wealth[x+1] + money
                    proportion[x+1] = proportion[x+1] + [strategies]
                wins.append(games.success)
                losses.append(games.failures)
                games.set_success(0)
                games.set_failures(0)
        agent_wealth = []
        low_error_wealth = []
        high_error_wealth = []
        successes = [wins[i:i+number_of_games] for i in range(0, len(wins), number_of_games)]
        failures = [losses[i:i+number_of_games] for i in range(0, len(losses), number_of_games)]
        for x in range(number_of_games+1):
            average_wealth = sum(wealth[x])/(len(wealth[x]))
            wealth_var = sum([((x - average_wealth) ** 2) for x in wealth[x]]) / len(wealth[x])
            wealth_std_dev = wealth_var ** 0.5
            high_confidence_interval = average_wealth + 1.96 * (wealth_std_dev/math.sqrt(len(wealth[x])))
            low_confidence_interval = average_wealth - 1.96 * (wealth_std_dev/math.sqrt(len(wealth[x])))
            agent_wealth.append(average_wealth)
            high_error_wealth.append(high_confidence_interval)
            low_error_wealth.append(low_confidence_interval)
        plt.errorbar(range(number_of_games+1), agent_wealth, yerr= np.asarray([low_error_wealth, high_error_wealth]), label = "average wealth of the players", errorevery=2)
        plt.legend()
        plt.xlabel('Number of games')
        plt.ylabel('Wealth')
        name = string + 'wealth.png'
        plt.savefig(name)
        plt.figure()
        all_strategies = ["Nothing", "Everything", "FairShare", "Tit-for-Tat", "Revenge"]
        for strat in all_strategies:
            line = []
            low_error_strat = []
            high_error_strat = []
            for y in range(number_of_games+1):
                average_strat = 0
                for x in proportion[y]:
                    average_strat += x.count(strat)
                average_strat_proportion = average_strat/10/games.pop_size
                average_strat = average_strat/10
                strat_var = 0
                for x in proportion[y]:
                    strat_var += (x.count(strat) - average_strat) ** 2
                strat_var = strat_var/10
                strat_std_dev = strat_var ** 0.5
                high_confidence_interval = (average_strat + 1.645 * (strat_std_dev / math.sqrt(10)))/games.pop_size
                low_confidence_interval = (average_strat - 1.645 * (strat_std_dev / math.sqrt(10)))/games.pop_size
                line.append(average_strat_proportion)
                high_error_strat.append(high_confidence_interval)
                low_error_strat.append(low_confidence_interval)
            plt.errorbar(range(number_of_games+1), line, yerr= np.asarray([low_error_strat, high_error_strat]), label= strat, errorevery=2)
        plt.legend()
        name = string + 'strategy.png'
        plt.xlabel('Number of games')
        plt.ylabel('Proportion of players using the strategy')
        plt.savefig(name)
        plt.figure(figsize=(9, 5))
        if punishment == 0.0:
            pun_0.append(sum(agent_wealth)/(number_of_games+1))
            coop_0.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.1:
            pun_1.append(sum(agent_wealth) / (number_of_games+1))
            coop_1.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.2:
            pun_2.append(sum(agent_wealth)/(number_of_games+1))
            coop_2.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.3:
            pun_3.append(sum(agent_wealth)/(number_of_games+1))
            coop_3.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.4:
            pun_4.append(sum(agent_wealth)/(number_of_games+1))
            coop_4.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.5:
            pun_5.append(sum(agent_wealth)/(number_of_games+1))
            coop_5.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.6:
            pun_6.append(sum(agent_wealth)/(number_of_games+1))
            coop_6.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.7:
            pun_7.append(sum(agent_wealth)/(number_of_games+1))
            coop_7.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.8:
            pun_8.append(sum(agent_wealth)/(number_of_games+1))
            coop_8.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 0.9:
            pun_9.append(sum(agent_wealth)/(number_of_games+1))
            coop_9.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        if punishment == 1:
            pun_10.append(sum(agent_wealth)/(number_of_games+1))
            coop_10.append(sum([x for x in np.mean(successes, axis=0)]) / len([x for x in np.mean(successes, axis=0)]))
        plt.bar(np.arange(number_of_games), [x for x in np.mean(failures, axis=0)], color = 'r', width = 0.5, align='edge' )
        plt.bar(np.arange(number_of_games)+0.5, [y for y in np.mean(successes, axis=0)], color = 'g', width = 0.5, align='edge')
        name = string + 'games.png'
        plt.xlabel('Number of games')
        plt.ylabel('Number of rounds that failed/succeed')
        plt.savefig(name)
        plt.close('all')
    heat = [pun_0, pun_1, pun_2, pun_3, pun_4, pun_5, pun_6, pun_7, pun_8, pun_9, pun_10]
    together = [coop_0, coop_1, coop_2, coop_3, coop_4, coop_5, coop_6, coop_7, coop_8, coop_9, coop_10]
    if rank != 0:
        comm.send([heat, together], dest=0)
    else:
        results = heat
        for worker in range(1, size):
            data = comm.recv(source=worker)
            i = 0
            while i < len(results):
                results[i] = results[i] + data[0][i]
                together[i] = together[i] + data[1][i]
                i += 1
        ax = sns.heatmap(results, xticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],yticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.set_xlabel('Fairness factor')
        ax.set_ylabel('Punishment amount')
        figure = ax.get_figure()
        figure.savefig('heatmap-hard_target.png')
        anova_wealth_file = open("anova_wealth.txt", "w")
        for row in results:
            np.savetxt(anova_wealth_file, row)
        anova_wealth_file.close()
        anova_win_file = open("anova_wins.txt", "w")
        for row in together:
            np.savetxt(anova_win_file, row)
        anova_win_file.close()
