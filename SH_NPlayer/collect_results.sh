#!/bin/sh

# concatenate all the reults from each MPI task into one file (results.csv)
# with header as first line
# WARNING: clobbers results.csv
# ADS 25Oct2012

if [ $# -ne 0 ]; then
  echo "usage: $0" >&2
  exit 1
fi

OUTFILE=results.csv

echo 'run_number, pop_size, pop_update_rule, pop_update_rate, enhance_factor, cost, effective_threshold, group_size, mutation_rate, cooperators, defectors, avReward, time' > $OUTFILE

cat results/result?.csv results/result??.csv results/result???.csv >> $OUTFILE


