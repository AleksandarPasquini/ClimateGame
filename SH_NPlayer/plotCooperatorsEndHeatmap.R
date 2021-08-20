#!/usr/bin/Rscript

# Plot heat maps (p_s--p_h phase space) of strategies at end of simulation
# (equilibrium for evolutionary game theory simulations.
# This verision usies the output from collect_results.sh collecting results
# from the model e.g. /lattice-jointactivity-simcoop-social-noise-cpp-end/model
# run by
# lattice-python-mpi/src/axelrod/geo/expphysicstimeline/multiruninitmain.py
# but without 'end' on the command line so writes stats at various time points
# rather than running to equilibrium and writing only at end, but takes
# last time point.
#
# The name of the CSV file with the data is given on stdin
#
# Usage:
#
# Rscript plotNumCooperatorsEndHeatmap.R data.csv outputfilenameprefix
#
# Output files are outputfilenamprefix-play_probXX-signal_costYY-detect_costZZ.eps
# and outputfilenamprefix-variance-play_probXX-signal_costYY-detect_costZZ.eps
#
# showing respectively the mean and variance of the strategy proportions,
#
# where XX is the play_prob value and YY is signal cost value an ZZ is signal
# detection cost
#  e.g. wellmixed_heatmap-play_prob0.5-signal_cost0.01-detect_cost0.00.eps
#       wellmixed_heatmap-variance-play_prob0.5-signal_cost0.01-detect_cost0.00.eps
#
# E.g. Rscript plotNumCooperatorsVsTimeMultiProb.R results.csv wellmixed_heatmap
#
# ADS February 2017


library(ggplot2)
library(doBy)
library(reshape)
library(scales)
library(gridExtra)
library(RColorBrewer)


bla <- function(variable, value) {

    # note use of bquote(), like a LISP backquote, evaluates only inside
    # .()
# http://stackoverflow.com/questions/4302367/concatenate-strings-and-expressions-in-a-plots-title
# but actually it didn't work, get wrong value for beta_s (2, even though no
   # such beta_s exists ?!? some problem with efaluation frame [couldn't get
# it to work by chaning where= arugment either), so using
# substitute () instead , which also doesn't work, gave up.
# --- turns out problem was I'd forgotten that all these vsariables have
# been converted to factors, so have to do .(levels(x)[x]) not just .(x)
    sapply      (value, FUN=function(xval ) 
        if (variable == "beta_s") {
          print(xval)
          bquote(beta[s]==.(levels(xval)[xval]))
        } else if (variable == "n") {
          bquote(N/L^2==.(levels(xval)[xval]))
        } else if (variable == "n_immutable") {
          bquote(F[I] == .(levels(xval)[xval]))
        } else if (variable == "m") {
          bquote(L == .(levels(xval)[xval]))
        } else if (variable == "payoff_r") {
          bquote(r == .(levels(xval)[xval]))
        } else if (variable == "play_prob") {
          bquote(p[play] == .(levels(xval)[xval]))
        } else if (variable == "mpcr") {
          bquote(plain("MPCR") == .(xval))
        } else if (variable == "punishment_fine") {
          bquote(alpha==.(levels(xval)[xval]))
        } else if (variable == "punisher_cost") {
          bquote(beta==.(levels(xval)[xval]))
        } else if (variable == "signal_cost") {
          bquote(c==.(levels(xval)[xval]))
        } else {
          bquote(.(variable) == .(levels(xval)[xval]))
        }
      )
}

responses <- c('cooperators')

#run_number, pop_size, pop_update_rule, pop_update_rate, enhance_factor, cost, effective_threshold, group_size, mutation_rate, cooperators, defectors, avReward, time

responsenames <- c('cooperators %')

stopifnot(length(responses) == length(responsenames))

if (length(commandArgs(trailingOnly=TRUE)) != 2) {
    cat("Usage: Rscript plotNumCooperatorsEndHeatmap.R results.csv outfilename.eps\n")
    quit(save='no')
}
results_filename <- commandArgs(trailingOnly=TRUE)[1]
output_prefix <- commandArgs(trailingOnly=TRUE)[2]


orig_experiment <- read.csv(results_filename, stringsAsFactors=F)

#orig_experiment$m <- as.factor(orig_experiment$m)
#orig_experiment$play_prob <- as.factor(orig_experiment$play_prob)
#orig_experiment$signal_cost <- as.factor(orig_experiment$signal_cost)

# get list of responses that were zero at start, do not plot these
initial_zero_responses <- NULL
start_values <- orig_experiment[which(orig_experiment$time == 0),]
for (response in responses) {
  if (all(is.na(start_values[,response])) ||
      all(start_values[,response] == 0)) {
    initial_zero_responses <- c(initial_zero_responses, response)
  }
}

# get only data et end of simulations
end_time <- max(orig_experiment$time)
orig_experiment <- orig_experiment[which(orig_experiment$time == end_time),]

orig_experiment[ ,enhance_factor] <- orig_experiment[ ,enhance_factor]/orig_experiment$group_size
orig_experiment[ ,effective_threshold] <- orig_experiment[ ,effective_threshold]/orig_experiment$group_size
orig_experiment[ ,group_size] <- orig_experiment[ ,group_size]/orig_experiment$pop_size

pop_size_list <- unique(orig_experiment$pop_size)
group_size_list <- unique(orig_experiment$group_size)
pop_update_rule_list <- unique(orig_experiment$pop_update_rule)
pop_update_rate_list <- unique(orig_experiment$pop_update_rate)

for (pop_size_i in 1:length(pop_size_list)) {
    for (group_size_i in 1:length(group_size_list)){
        #for (pop_update_rule_i in 1:length(pop_update_rule_list)){
            for (pop_update_rate_i in 1:length(pop_update_rate_list)){
                experiment <- orig_experiment
                experiment <- experiment[which(experiment$pop_size == pop_size_list[pop_size_i]),]
                experiment <- experiment[which(experiment$group_size == group_size_list[group_size_i]),]
                #experiment <- experiment[which(experiment$pop_update_rule == pop_update_rule_list[pop_update_rule_i]),]
                experiment <- experiment[which(experiment$pop_update_rate == pop_update_rate_list[pop_update_rate_i]),]

                if (nrow(experiment) == 0) {
                    print(paste('skipping pop_size = ', pop_size_list[pop_size_i],
                    'group_size = ', group_size_list[group_size_i],
                    #'pop_update_rule = ', pop_update_rule_list[pop_update_rule_i],
                    'pop_update_rate = ', pop_update_rate_list[pop_update_rate_i],
                    'not in data',)
                    )
                    next
                }
    

                #D<-melt(experiment, id=c('n','m','F','q','radius','payoff_r','play_prob','hare_prob_1','hare_prob_2','stag_prob','signal_detection_cost','signal_cost','signal_control_cost', 'dispositionist_cost', 'intentionist_cost', 'delta_d', 'delta_m', 'mistake_detection_probability', 'weight_threshold', 'stranger_interaction_probability', 'punish_factor', 'punish_threshold_pool', 'run','time') )
                D<-melt(experiment, id=c('run_number', 'pop_size', 'pop_update_rule', 'pop_update_rate', 'enhance_factor', 'cost', 'effective_threshold', 'group_size', 'mutation_rate',  'time') )
                #D<-summaryBy(value ~ n + m + F + q + radius + payoff_r + play_prob + hare_prob_1 + hare_prob_2 + stag_prob + signal_detection_cost + signal_cost + signal_control_cost + dispositionist_cost + intentionist_cost + delta_d + delta_m + mistake_detection_probability + weight_threshold + stranger_interaction_probability + punish_factor + punish_threshold_pool + time + variable, data=D, FUN=c(mean, sd, var))
                D<-summaryBy(value ~ run_number + pop_size + pop_update_rule + pop_update_rate + enhance_factor + cost + effective_threshold + group_size + mutation_rate  + time + variable, data=D, FUN=c(mean, sd, var))
                
    
    
                # plot 0 Sum
                plotlist0 <- list()
                varplotlist0 <- list()
                #for (i in 1:length(responses)) {
                    #response <- responses[i]
                    #responsename <- responsenames[i]
                for (i in 1:length(pop_update_rule_list)) {
                      response <- responses[1]
                      responsename <- pop_update_rule_list[i]
                      if (!(response %in% colnames(experiment))) {
                        print(paste('skipping response ', response, ' not in data'))
                        next
                      }
                      Dst <- D[which(D$variable == response), ]
                      Dst <- Dst[which(Dst$pop_update_rule == pop_update_rule_list[i]), ]

                      print(response) #XXX
                      if (all(is.na(Dst$value.mean))) {
                        print(paste('skipping response ', response, ' all values are NA'))
                        next
                      }
                      if (response %in% initial_zero_responses) {
                        print(paste('skipping  ', response, ' all values were initially zero'))
                        next
                      }
                      
                      p <- ggplot(Dst, aes(enhance_factor, effective_threshold))
                      p <- p + xlab(expression(F))
                      p <- p + ylab(expression(M))
                      p <- p + geom_raster(aes(fill = value.mean) )
                      #p <- p + coord_fixed(ratio = 1)
                      p <- p + ggtitle(responsename)
                      
                      #p <- p + scale_fill_distiller(bquote(p_[.(responsename)]), limits=c(0,1))
                      #p <- p + scale_fill_distiller("", palette="Spectral", limits=c(0,1), guide="colourbar")
                      #p <- p + scale_fill_continuous("", limits=c(0,1), guide="colourbar")
                      p <- p + scale_fill_continuous("", limits=c(0,1), low="white", high="black",
                          guide="colourbar" )

                      p <- p + theme_bw() +
                              theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
                      p <- p  + theme(plot.title = element_text(size=10))
                      plotlist0 <- c(plotlist0, list(p))

                      #
                      # variance heatmaps
                      #
                      varp <- ggplot(Dst, aes(enhance_factor, effective_threshold))
                      varp <- varp + xlab(expression(F))
                      varp <- varp + ylab(expression(M))
                      varp <- varp + geom_raster(aes(fill = value.sd) )
                      #varp <- varp + coord_fixed(ratio = 1)
                      varp <- varp + ggtitle(responsename)
                      varp <- varp + scale_fill_continuous("", low="white", high="black",
                          guide="colourbar" )

                      varp <- varp + theme_bw() +
                              theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
                      varp <- varp  + theme(plot.title = element_text(size=10))
                      varplotlist0 <- c(varplotlist0, list(varp))
                }
                # EPS suitable for inserting into LaTeX
                outfilename0 <- paste(output_prefix, '-sum', '-pop_size_', pop_size_list[pop_size_i],
                                      '-group_size_', group_size_list[group_size_i],
                                      #'-pop_update_rule_', pop_update_rule_list[pop_update_rule_i],
                                      '-pop_update_rate_', pop_update_rate_list[pop_update_rate_i],
                    '.eps', sep='')
                print(outfilename0)#XXX
                varoutfilename0 <- paste(output_prefix, '-sum', '-variance',
                                      '-pop_size_', pop_size_list[pop_size_i],
                                      '-group_size_', group_size_list[group_size_i],
                                      #'-pop_update_rule_', pop_update_rule_list[pop_update_rule_i],
                                      '-pop_update_rate_', pop_update_rate_list[pop_update_rate_i],
                    '.eps', sep='')
                    
                postscript(outfilename0,
                      onefile=FALSE,paper="special",horizontal=FALSE,
                      width = 9, height = 6)
                do.call(grid.arrange, plotlist0)
                dev.off()

                postscript(varoutfilename0,
                      onefile=FALSE,paper="special",horizontal=FALSE,
                      width = 9, height = 6)
                do.call(grid.arrange, varplotlist0)
                dev.off()
            }
        }
    }
}
    

