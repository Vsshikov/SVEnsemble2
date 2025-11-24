#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

module load gatk samtools

orig_coverage=300
downsample_factor=0.1

downsampled_coverage=$(echo "$orig_coverage * $downsample_factor" | bc)
path_to_300x="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878/aligned/NA12878.minimap.bam"
output_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878_${downsampled_coverage}x/aligned"

input_basename=$(basename "$path_to_300x")
output_bam="${output_dir}/${input_basename%.minimap.bam}_${downsampled_coverage}x.minimap.bam"

echo "Downsampling $path_to_300x to $output_bam"

gatk DownsampleSam -I "$path_to_300x" -O "$output_bam" -P $downsample_factor
