#!/bin/sh

#D="Fashion-MNIST"
D="SIFT"

for D in "SIFT" "Fashion-MNIST"
do


DS="../data/$D/"

for Q in 10 2000 10000
do

  export NUM_QUERIES=$Q

  F=tests-$D-$Q.txt
  R=${F}-results.txt

  : > $F
  echo "Searching ${DS} queries=$Q ... output file: $F"

  echo "Searching dataset $D queries=$Q" > $R
  echo "============================================" >> $R
  


  for K in 1 4 5 8 10 20
  do 

    echo "... Searching ${DS} queries=$Q k=$K"

    for S in 0 0.3 0.31 0.32 0.4 0.5 0.7 1.0 1.5
    do

      echo "=== Searching ${DS} queries=$Q k=$K m=$S ===" >> $F

      export STAT_FILTER_TOPK=$K
      export SCALE_OVERRIDE=$S

      target/release/vector_search_evaluator ${DS} >> $F

    done
  done

  rm -f ${F}_tmpa ${F}_tmpb ${F}_tmpc
  grep 'Recall rate: ' $F | awk '{print $3}' > ${F}_tmpa
  grep ' time:  ' $F | awk '{print $8,$9}' > ${F}_tmpb
  grep ' Searching ' $F > ${F}_tmpc
  paste ${F}_tmpa ${F}_tmpb ${F}_tmpc | sort -n -r >> $R
  rm -f ${F}_tmpa ${F}_tmpb ${F}_tmpc

  echo "Results in $R"
  head -12 $R

done


done
