#!/bin/bash
#PBS -u macias
#PBS -N vib2_0.00
#PBS -l nodes=qcexnod57:ppn=2
#PBS -S /bin/bash
FILE="vib2_0.00"
export ScrDir=/scr/macias/${PBS_JOBID}_$FILE
mkdir -p $ScrDir
Wdir="/home/macias/bin/anhdis/anhdis2review/vib2/0.00"
. /soft/g16.a03/g16/bsd/g16.profile
cd $ScrDir
g16 < $Wdir/$FILE.com > $ScrDir/$FILE.log
cp *.log $Wdir
cp *.o $Wdir
cp *.e $Wdir
exit
