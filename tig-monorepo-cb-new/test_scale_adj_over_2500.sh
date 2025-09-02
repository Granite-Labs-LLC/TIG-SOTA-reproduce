#!/bin/bash


#echo "SCALE_ADJUSTMENT"
#
NQ=2000
NN=10

ST=0

DIFFICULTIES=('[2500,625]' '[2600,650]' '[2700,675]' '[2800,700]' '[2900,725]' '[3000,750]')
#DIFFICULTIES=('[200,50]' '[300,75]' '[400,100]' '[500,125]' '[700,175]' '[852,213]' '[1000,250]' '[1200,300]' '[1420,355]' '[1640,410]' '[1880,470]' '[2000,500]' '[2100,525]' '[2200,550]' '[2336,584]' '[2400,600]' '[2500,625]' '[2600,650]' '[2700,675]' '[2800,700]' '[2900,725]' '[3000,750]')

F=tmpfile1

: > $F

echo "start_nonce,scale_adjustment,queries_per_nonce,queries_per_second,recall_percent,actual_scale_factor"

for DIFFICULTY in "${DIFFICULTIES[@]}"; do

 for ST in 0 2348587; do

  #for S in 0.6 0.65 0.66 0.67 0.68 0.69 0.7 0.74 0.78 0.8 1.0
  #for S in 0.4 0.5 0.6 0.65 0.66 0.67 0.68 0.69 0.7 0.74 0.78 0.8 1.0 1.1 1.2 1.5 1.8; do
  #for S in 1.0 1.1 1.2 1.5 1.8; do
  #for S in 0.05 0.08 0.1 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.65 0.66 0.67 0.68 0.69 0.7 0.74 0.78 0.8 1.0 1.1 1.2 1.5 1.8; do
  S=1.0
  for S in 1.0 1.1 1.2 1.5 1.8 2.0 2.2 2.4 2.6 2.8 3.0; do
  #echo "scale adjustment: $S"


    export SCALE_ADJUSTMENT=$S

    NQ=$(echo "$DIFFICULTY" | tr -d '[]' | cut -f1 -d ',')

    scripts/test_algorithm_timer --start ${ST} --nonces ${NN} --workers 1 stat_filter ${DIFFICULTY} | egrep 'Time for |#solutions: |scale_applied:' > $F 

    T=`grep 'Time for ' $F | awk '{printf("%.3f + ", $4)}'`
    ms_per_nonce=$(echo "scale=3; ($T 0) / $NN" | bc)
    #echo $ms_per_nonce
    qps=$(echo "scale=4; (1000.0 / $ms_per_nonce) * ${NQ}.0" | bc | cut -f1 -d'.')

    N=`grep '#solutions: ' $F | sed -e '/^.*#solutions: /s///' | cut -f1 -d',' | tail -1`

    percent=$(echo "scale=2; $N / $NN" | bc)
    #percent=$(( N * 100 / NN ))
    #echo "${percent}%"

    A=`grep 'scale_applied:' $F | cut -f2 -d':' | cut -f2 -d',' | tail -1`

    echo "${ST},$S,${NQ},${qps},${percent},$A"

    # Enable this if looping through scale adjustments
    [ "$percent" = "1.00" ] && break

  done
 done
done

