#!/bin/bash
#SBATCH --job-name=ift_llama
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --partition=XXXXX
#SBATCH --gres=gpu:8 # Adjust number of GPUs here
#SBATCH --output=shared_storage/sourab/temp/logs/%x-%j.out
#SBATCH --err=shared_storage/sourab/temp/logs/%x-%j.err

set -x -e

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
source ~/sourab/.bashrc
source shared_storage/sourab/miniconda3/etc/profile.d/conda.sh
conda activate hf
cd shared_storage/sourab/DHS-LLM-Workshop/code_assistant/training
git pull

# have the below in case of debugging nccl issues such as nccl timeout.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file configs/fsdp_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="\
train.py \
    --model_name "codellama/CodeLlama-7b-hf" \
    --dataset_name "smangrul/code-chat-assistant-v1" \
    --max_seq_len 2048 \
    --max_steps 500 \
    --logging_steps 25 \
    --eval_steps 100 \
    --save_steps 250 \
    --bf16 True \
    --packing True \
    --output_dir "/shared_storage/sourab/experiments/full-finetune-llama-chat-asst" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataset_text_field "content" \
    --use_gradient_checkpointing True \
    --learning_rate 5e-5  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --use_flash_attn True
"


export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
