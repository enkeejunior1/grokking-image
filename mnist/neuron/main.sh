#!/bin/bash
#SBATCH --job-name=grok
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# slurm_path="/home/yonghyun.park/slurm_dit.sif" 

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate iclr
cd $SLURM_SUBMIT_DIR

# Actual work
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# python classifier.py
# python train.py --train_fraction 0.9
# for p in 0.9 0.7 0.5 0.3 0.1; do
#     python train_2.py --train_fraction $p &
# done

# recommend setting: train_fraction 0.7, num_images 1, 4, 16, 64, 256, 1024, 4096
# num_images=1
# python train.py --train_fraction 0.9 --num_images $num_images
# for i in 0 1 2 3 4 5 6 7 8 9; do
#     python analyze_ffn_neurons.py --model_path weights/train_fraction_0.9-num_images_1/model_final.pth --train_fraction 1.0 --num_images 1 --target_class $i
# done

# python analyze_ffn_pca.py --model_path weights/train_fraction_0.9-num_images_1/model_final.pth --train_fraction 1.0 --num_images 1
# python analyze_ffn_knn.py --model_path weights/train_fraction_0.9-num_images_1/model_final.pth --train_fraction 1.0 --num_images 1
# python analyze_knn_global.py --model_path weights/train_fraction_0.9-num_images_1/model_final.pth --train_fraction 1.0 --num_images 1
python analyze_pca.py --model_path weights/train_fraction_0.9-num_images_1/model_final.pth --train_fraction 1.0 --num_images 1