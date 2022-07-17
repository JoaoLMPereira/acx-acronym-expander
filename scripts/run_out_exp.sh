#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
benchmark_script_path=$1
out_expanders_file_path=$2


while IFS='' read -r line || [[ -n "$line" ]]; do
    #echo "Text read from file: $line2"
    ARGS_C=$(echo $line |sed 's/[ \t]+*/ /g' | sed 's/['\'']+*/"/g')

    out_expander=$(echo $ARGS_C |cut -d' ' -f 1)
    
    out_expander_args=$(echo $ARGS_C |cut -d' ' -f 2-)
    
    if [ "$out_expander" == "$out_expander_args" ]; then
        echo $benchmark_script_path --save-and-load --out_expander $out_expander
        python $benchmark_script_path --save-and-load --out_expander $out_expander
    else

        echo $benchmark_script_path --save-and-load --out_expander $out_expander --out_expander_args "${out_expander_args}"
        python $benchmark_script_path --save-and-load --out_expander $out_expander --out_expander_args "${out_expander_args}"
    fi

done < "$out_expanders_file_path"
