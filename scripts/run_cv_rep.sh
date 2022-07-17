#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
benchmark_script_path=$1
representator=$2

if [ -z "$3" ]
  then
  start_rep_exp=1
  else
  start_rep_exp=$3
fi

rep_file="${DIR}"\/"${representator}_exper.csv"


while IFS='' read -r line2 || [[ -n "$line2" ]]; do
    #echo "Text read from file: $line2"
    REP_ARGS_C=$(echo $line2 |sed 's/[ \t]+*/,/g')
    # Removes first number which is a parameters combination ID
    exp_rep_num=$(echo $REP_ARGS_C |cut -d',' -f 1)
    if [ $exp_rep_num -lt  $start_rep_exp ]
        then
        continue
    fi
        
    echo "experiment numbers: representator= "$exp_rep_num
    REP_ARGS=$(echo $REP_ARGS_C |cut -d',' -f 2-)

    JSON_ARGS="[\"${representator}\",[${REP_ARGS} ]]"
    echo "out_expander_args: "$JSON_ARGS
        
    python $benchmark_script_path --crossvalidation --out_expander cossim --out_expander_args "${JSON_ARGS}"
        
done < "$rep_file"
