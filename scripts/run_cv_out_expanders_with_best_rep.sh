#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
benchmark_script_path=$1
out_expander=$2
best_rep_file=$3

if [ -z "$4" ]
  then
  start_out_exp=1
  else
  start_out_exp=$4
fi

if [ -z "$5" ]
  then
  start_rep_exp=1
  else
  start_rep_exp=$5
fi

exp_file="${DIR}"\/"${out_expander}_exper.csv"


while IFS='' read -r line2 || [[ -n "$line2" ]]; do
    #echo "Text read from file: $line2"
    REP_ARGS_C=$(echo $line2 |sed 's/[ \t]+*/ /g' | sed 's/['\'']+*/"/g')
    # Removes first number which is a parameters combination ID
    representator=$(echo $REP_ARGS_C |cut -d' ' -f 1)
    
        
    echo "representator= "$representator
    REP_ARGS=$(echo $REP_ARGS_C |cut -d' ' -f 2-)
    
    
    while IFS='' read -r line || [[ -n "$line" ]]; do
        #echo "Text read from file: $line"
        ARGS_C=$(echo $line |sed 's/[ \t]+*/,/g')
        exp_out_num=$(echo $ARGS_C |cut -d',' -f 1)
        #echo $exp_out_num
        if [ $exp_out_num -lt  $start_out_exp ]
        then
            continue
        fi
        echo "out expander experiment number=" $exp_out_num
        ARGS=$(echo $ARGS_C |cut -d',' -f 2-)


        if [ "$representator" == "$REP_ARGS" ]; then
            JSON_ARGS="[${ARGS},\"${representator}\" ]"
        else
            JSON_ARGS="[${ARGS},\"${representator}\",${REP_ARGS} ]"
        fi
        
        echo "out_expander_args: "$JSON_ARGS
        
        python $benchmark_script_path --crossvalidation --save-and-load --out_expander $out_expander --out_expander_args "${JSON_ARGS}"
    done < "$exp_file"
done < "$best_rep_file"

sh after_run_cv_out_expanders_with_best_rep.sh
