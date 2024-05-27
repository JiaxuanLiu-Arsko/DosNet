#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --partition=h800
#SBATCH --job-name=bs_learning
#SBATCH -A hmt03
#SBATCH --output=./log
#SBATCH --error=./err ### 错误日志文件.

NP=$((1)) ### 使用的 gpu 卡数，与--gres=gpu:1 的数字一致
### 执行任务所需要加载的模块
module load oneapi22.3
module load nvhpc/22.11
module load cuda11.8
### 一些提示性输出
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MV2_ENABLE_AFFINITY=0
export NCCL_P2P_LEVEL=NVL
echo ”The current job ID is $SLURM_JOB_ID”
echo ”Running on $SLURM_JOB_NUM_NODES nodes:”
echo $SLURM_JOB_NODELIST
echo ”Using $SLURM_NTASKS_PER_NODE tasks per node”
echo ”A total of $SLURM_NTASKS tasks is used”
### 对任务执行的内存不做限制
ulimit -s unlimited
ulimit -c unlimited
### 加载任务所需要的库
export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
echo $LD_LIBRARY_PATH

###防止conda activate environment失败，网上抄的
if [ -f "/home/jxliu/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/jxliu/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/jxliu/anaconda3/bin/:$PATH"
fi

### 执行任务
conda activate nequIP
python _from_poscar_to_dos.py --poscar ./test/mp-8468.vasp --model ./test/model_36.pth --output ./test\
 --config ./test/model_config.yaml