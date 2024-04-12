#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp22          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu          # Select the ai partition
#SBATCH --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=abis28891@gmail.com  # Email address for notifications

python train.py --normalize --gat_model
python test.py --data_path ../../tsp_n5900/test2.txt --model_path ../../checkpoint/version0/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version0/run_results --output_path ../../checkpoint/version0/test_results --use_gpu --tsp
python train.py --normalize --dataset_directory ../../atsp_n5900/ --conv_type sage
python test.py --data_path ../../atsp_n5900/test2.txt --model_path ../../checkpoint/version1/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version1/run_results --output_path ../../checkpoint/version1/test_results --use_gpu
python train.py --normalize --dataset_directory ../../atsp_n5900/ --learn_alpha --conv_type gcn
python test.py --data_path ../../atsp_n5900/test2.txt --model_path ../../checkpoint/version2/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version2/run_results --output_path ../../checkpoint/version2/test_results --use_gpu
python train.py --normalize --dataset_directory ../../atsp_n5900/ --learn_alpha --conv_type gat
python test.py --data_path ../../atsp_n5900/test2.txt --model_path ../../checkpoint/version3/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version3/run_results --output_path ../../checkpoint/version3/test_results --use_gpu
python train.py --normalize --dataset_directory ../../atsp_n5900/ --learn_alpha --conv_type dir-sage
python test.py --data_path ../../atsp_n5900/test2.txt --model_path ../../checkpoint/version4/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version4/run_results --output_path ../../checkpoint/version4/test_results --use_gpu
python train.py --normalize --dataset_directory ../../atsp_n5900/ --learn_alpha --conv_type dir-gcn
python test.py --data_path ../../atsp_n5900/test2.txt --model_path ../../checkpoint/version5/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version5/run_results --output_path ../../checkpoint/version5/test_results --use_gpu
python train.py --normalize --dataset_directory ../../atsp_n5900/ --learn_alpha --conv_type dir-gat
python test.py --data_path ../../atsp_n5900/test2.txt --model_path ../../checkpoint/version6/checkpoint_best_val.pt --guides regret_pred --run_dir  ../../checkpoint/version6/run_results --output_path ../../checkpoint/version6/test_results --use_gpu
