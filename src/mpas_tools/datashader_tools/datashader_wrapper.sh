#!/bin/bash
#$ -M cdn5285@psu.edu
#$ -m abe
#$ -r y

#PBS -l walltime=02:00:00
#PBS -A cmz5202_a_g_sc_default
#PBS -l nodes=4

cd /storage/work/cnd5285/github/mpas_tools/mpas_tools/datashader

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
    conda run -n datashader_tools python datashader_mpas.py -ts $i -tv 'FLUT' -ti '0831 TS ${i} PR 10' -pr 10
    
for i in 1 5 6 7 8 9 10 11 12 13 14 15 20 25 50 100
do
    conda run -n datashader_tools python datashader_mpas.py -ts 12 -tv 'FLUT' -ti '0831 TS 12 PR ${i}' -pr $i

