#!/bin/bash -l

#PBS -l nodes=4:ppn=20
#PBS -l walltime=2:00:00
#PBS -A cmz5202_a_g_sc_default
#PBS -j oe
#PBS -N TE.track

### Navigate to my script directly
cd /storage/home/${LOGNAME}/tempest-scripts/tgw/

### Load required modules
module purge
module load gcc/8.3.1
module load openmpi/3.1.6
module load netcdf/4.4.1

############ USER OPTIONS #####################

## Unique string (useful for processing multiple data sets in same folder
UQSTR=ERA5

## Path to TempestExtremes binaries on YS
TEMPESTEXTREMESDIR=/storage/home/cmz5202/sw/tempestextremes/

## Topography filter file (needs to be on same grid as PSL, U, V, etc. data
TOPOFILE=/gpfs/group/cmz5202/default/h1files/topo/${UQSTR}.topo.nc

## If using unstructured CAM-SE ne120 data
CONNECTFLAG="" 

## List of years to process
YEARSTRARR=`seq 1980 2018`

## Path where files are
PATHTOFILES=/gpfs/group/cmz5202/default/h1files/${UQSTR}/

############ TRACKER MECHANICS #####################
starttime=$(date -u +"%s")

## DATESTRING gives us a unique integer for files in the script
DATESTRING=`date +"%s%N"`
FILELISTNAME=filelist.txt.${DATESTRING}
TRAJFILENAME=trajectories.txt.${UQSTR}
touch $FILELISTNAME

## Loop over all years, collect files, add them to our input file list
for zz in ${YEARSTRARR}
do
  find ${PATHTOFILES} -name "*h1.${zz}????.nc" | sort -n >> $FILELISTNAME
done
# Add static file(s) to each line
sed -e 's?$?;'"${TOPOFILE}"'?' -i $FILELISTNAME

## Define our heuristic settings/rules
DCU_PSLFOMAG=200.0
DCU_PSLFODIST=5.5
DCU_WCFOMAG=-6.0    # Z300Z500 -6.0, T400 -0.4
DCU_WCFODIST=6.5
DCU_WCMAXOFFSET=1.0
DCU_WCVAR="_DIFF(Z300,Z500)"   #DCU_WCVAR generally _DIFF(Z300,Z500) or T400
DCU_MERGEDIST=6.0
SN_TRAJRANGE=8.0
SN_TRAJMINLENGTH=10
SN_TRAJMAXGAP=3
SN_MAXTOPO=150.0
SN_MAXLAT=50.0
SN_MINWIND=10.0
SN_MINLEN=10

NCPU=80
STRDETECT="--verbosity 0 --timestride 1 ${CONNECTFLAG} 
--out cyclones_tempest.${DATESTRING} 
--closedcontourcmd PSL,${DCU_PSLFOMAG},${DCU_PSLFODIST},0;${DCU_WCVAR},${DCU_WCFOMAG},${DCU_WCFODIST},${DCU_WCMAXOFFSET} 
--mergedist ${DCU_MERGEDIST} 
--searchbymin PSL 
--outputcmd PSL,min,0;_VECMAG(UBOT,VBOT),max,2;PHIS,max,0"
echo $STRDETECT
touch cyclones.${DATESTRING}

## Run the "detect" stage in parallel using the settings from STRDETECT above.
mpirun --np ${NCPU} --hostfile $PBS_NODEFILE  ${TEMPESTEXTREMESDIR}/bin/DetectNodes --in_data_list "${FILELISTNAME}" ${STRDETECT} </dev/null
# okay so --np 80 is the number of parallel MPI processes (4 nodes x 20 processors per node)
# --hostfile is specific to ROAR, it's a file where all relevant node hostnames are stored for a job
# ./DetectNodes 


## Glue individual cyclone files together into one big candidate file
cat cyclones_tempest.${DATESTRING}* >> cyclones.${DATESTRING}

## Remove the individual cyclone files, which we don't need anymore.
rm cyclones_tempest.${DATESTRING}*

## Stitch candidate cyclones together
${TEMPESTEXTREMESDIR}/bin/StitchNodes --out_file_format "csv" --format "i,j,lon,lat,slp,wind,phis" --range ${SN_TRAJRANGE} --minlength ${SN_TRAJMINLENGTH} --maxgap ${SN_TRAJMAXGAP} --in cyclones.${DATESTRING} --out ${TRAJFILENAME} --threshold "wind,>=,${SN_MINWIND},${SN_MINLEN};lat,<=,${SN_MAXLAT},${SN_MINLEN};lat,>=,-${SN_MAXLAT},${SN_MINLEN};phis,<=,${SN_MAXTOPO},${SN_MINLEN}"

echo "Cleaning up!"
rm ${FILELISTNAME}
rm log*.txt
rm cyclones.${DATESTRING}   #Delete candidate cyclone file

## Print some timing stats
endtime=$(date -u +"%s")
tottime=$(($endtime-$starttime))
printf "${tottime},reanalysis_track,${UQSTR}\n" >> timing.txt
