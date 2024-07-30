#!/bin/bash

# requires seqkit to be installed

conda activate gridss

input_fa="/home/users/allstaff/shikov.v/projects/SVensemble2/data/hg38/downloads/hg38.fa"
output_fa=$(echo $input_fa | sed -e "s/.fa/.no_decoys.fa/g")

input_fai=$(echo $input_fa | sed -e "s/.fa/.fa.fai/g")
output_fai=$(echo $input_fa | sed -e "s/.fa/.no_decoys.fa.fai/g")


seqkit grep -r -n -p 'chr([0-9]|[0-9][0-9]|M|X|Y)$' $input_fa > $output_fa
grep -P "chr([0-9]|[0-9][0-9]|M)\t" $input_fai > $output_fai
