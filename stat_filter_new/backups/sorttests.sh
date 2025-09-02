#!/bin/sh

F=$1

grep 'Recall rate: ' $F | awk '{print $3}' > ${F}_tmpa
grep ' time:  ' $F | awk '{print $8,$9}' > ${F}_tmpb
paste ${F}_tmpa ${F}_tmpb > ${F}_tmpc
sort -n -r ${F}_tmpc > ${F}_sorted.txt

echo "Results in ${F}_sorted.txt"
