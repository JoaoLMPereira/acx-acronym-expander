#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
benchmark_script_path=$1

if [ -z "$2" ]
  then
  start_rep_exp=1
  else
  start_rep_exp=$2
fi


representators='doc2vec tfidf lda' #lda_exp doc2vec_exp locality ngrams_contextvector?

for item in $representators;
do 
    echo "Representator: " $item
    sh ./scripts/run_cv_rep.sh $1 $item $start_rep_exp;
done

sh after_run_cp_representators.sh
