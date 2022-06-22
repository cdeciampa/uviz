#!/bin/bash
#$ -M cdn5285@psu.edu
#$ -m abe
#$ -r y

USERACCT=cmz5202_a_g_sc_default     # Run on our dedicated account.
JOBQUEUE=batch                      # Run in the batch queue.
JOBWALLCLOCK="00:58:00"             # Run for max 58 minutes.
NUMNODES=-4                         # Run on 4 nodes.

conda activate datashader
cd /storage/work/cnd5285/github/mpas_tools/mpas_tools/datashader

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
    python datashader_mpas.py -ts $i -tv 'FLUT' -ti '0830 TS ${i} PR 1'
    
for i in 1 10 50 100
do
    python datashader_mpas.py -ts 20 -tv 'FLUT' -ti '0830 TS 20 PR ${i}' 

