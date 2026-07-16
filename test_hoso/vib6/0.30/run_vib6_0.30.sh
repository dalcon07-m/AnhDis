#!/bin/bash
#PBS -u macias
#PBS -N vib6_0.30
#PBS -l nodes=qcexnod57:ppn=2
#PBS -S /bin/bash
FILE="vib6_0.30"
export ScrDir=/scr/macias/${PBS_JOBID}_$FILE
mkdir -p $ScrDir
Wdir="/home/macias/bin/anhdis/anhdis2review/vib6/0.30"
. /soft/g16.a03/g16/bsd/g16.profile
cd $ScrDir
g16 < $Wdir/$FILE.com > $ScrDir/$FILE.log
cp *.log $Wdir
cp *.o $Wdir
cp *.e $Wdir
exit
