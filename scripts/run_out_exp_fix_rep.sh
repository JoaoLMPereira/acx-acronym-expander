#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
benchmark_script_path=$1
out_expander=$2
representator=$3
REP_ARGS=${@:4}
exp_file="${DIR}"\/"${out_expander}_exper.csv"

while IFS='' read -r line || [[ -n "$line" ]]; do
    echo "Text read from file: $line"
    # Removes first number which is a parameters combination ID
    ARGS=$(echo $line | sed "s/^[^[[:space:]]]*[[:space:]]* //")
    echo "$ARGS"
    REP_ARGS_ESCAPED=$(printf "%q" "$REP_ARGS")
    echo "$2 $REP_ARGS_ESCAPED"
    EXPANDER_ARGS_ESCAPED=$(printf "%q" "$representator $REP_ARGS_ESCAPED")
    python $benchmark_script_path -cv $out_expander "$ARGS $representator $REP_ARGS_ESCAPED"
    exit
done < "$exp_file"

#sh after_run_rep.sh
