#!/bin/bash
#PBS -u macias
#PBS -N vib6_0.40
#PBS -l nodes=qcexnod57:ppn=2
#PBS -S /bin/bash
FILE="vib6_0.40"
export ScrDir=/scr/macias/${PBS_JOBID}_$FILE
mkdir -p $ScrDir
Wdir="/home/macias/bin/anhdis/anhdis2review/vib6/0.40"
. /soft/g16.a03/g16/bsd/g16.profile
cd $ScrDir
g16 < $Wdir/$FILE.com > $ScrDir/$FILE.log
cp *.log $Wdir
cp *.o $Wdir
cp *.e $Wdir
exit
