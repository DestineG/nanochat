#!/bin/bash

# =============================================================================
# 1. 参数配置获取 (交互式)
# =============================================================================

# 获取 NANOCHAT_BASE_DIR
default_base_dir="$HOME/.cache/nanochat"
echo "--------------------------------------------------"
echo "请输入项目中间产物目录 (直接回车使用默认值):"
read -p "路径 [$default_base_dir]: " input_dir
NANOCHAT_BASE_DIR=${input_dir:-$default_base_dir}

# 检测目录是否存在
if [ ! -d "$NANOCHAT_BASE_DIR" ]; then
    echo "目录 $NANOCHAT_BASE_DIR 不存在"
    read -p "是否创建该目录？(y/n) [y]: " mkdir_choice
    mkdir_choice=${mkdir_choice:-y} 
    
    if [[ "$mkdir_choice" == [yY] ]]; then
        mkdir -p "$NANOCHAT_BASE_DIR"
        if [ $? -eq 0 ]; then
            echo "目录创建成功"
        else
            echo "目录创建失败，请检查权限"
            exit 1
        fi
    else
        echo "未创建目录，脚本退出"
        exit 1
    fi
else
    echo "目录已存在，继续执行"
fi

echo
# 设置 GPU 数量
echo "--------------------------------------------------"
echo "检测到系统中可用 GPU 如下："
nvidia-smi -L 2>/dev/null || echo "未检测到 NVIDIA GPU"
echo
default_nproc=1
read -p "请输入要使用的 GPU 数量 (NPROC_PER_NODE) [$default_nproc]: " nproc_input
NPROC_PER_NODE=${nproc_input:-$default_nproc}
echo "设定 NPROC_PER_NODE 为: $NPROC_PER_NODE"

echo
# FP8 选项
echo "==================== 支持 FP8 的 NVIDIA GPU ===================="
echo "【消费级】 RTX 40/50 系列 | 【专业级】 H100/H200/L40S/RTX Ada"
echo "=================================================================="
read -p "是否启用 --fp8 训练？(y/n) [n]: " fp8_choice
fp8_choice=${fp8_choice:-n} 

fp8_arg=""
if [[ "$fp8_choice" == [yY] ]]; then
    fp8_arg="--fp8"
    echo "已启用：$fp8_arg"
else
    echo "不启用 FP8"
fi

echo
# Batch Size 选项
echo "--------------------------------------------------"
default_batch_size=1
read -p "请输入每个设备的 Batch Size (device-batch-size) [$default_batch_size]: " batch_input
DEVICE_BATCH_SIZE=${batch_input:-$default_batch_size}
echo "设定 Device Batch Size 为: $DEVICE_BATCH_SIZE"

echo
# WANDB 运行名称
echo "--------------------------------------------------"
echo "请输入 Weights & Biases 运行名称 (直接回车跳过日志记录):"
read -p "WANDB_RUN [dummy]: " wandb_input
WANDB_RUN=${wandb_input:-dummy}

# -----------------------------------------------------------------------------
# 2. 导出环境变量
# -----------------------------------------------------------------------------
export NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_HUB_CACHE=true
export HF_HUB_CACHE="$NANOCHAT_BASE_DIR/hf_hub_cache"
export HF_DATASETS_CACHE="$NANOCHAT_BASE_DIR/hf_datasets_cache"
export WANDB_RUN="$WANDB_RUN"

echo
echo "==================== 配置汇总 ===================="
echo "BASE_DIR: $NANOCHAT_BASE_DIR"
echo "GPU 数量 : $NPROC_PER_NODE"
echo "FP8 选项 : ${fp8_arg:-已关闭}"
echo "Device Batch Size : $DEVICE_BATCH_SIZE"
echo "W&B 运行 : $WANDB_RUN"
echo "=================================================="
echo

# -----------------------------------------------------------------------------
# 环境搭建

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 日志系统设置
# 如果使用 wandb 进行日志记录：
# 1) 在环境变量中设置 `WANDB_API_KEY`
# 2) 在本地先运行 `wandb login`
# 3) 运行脚本时设置 WANDB_RUN 环境变量，例如：`WANDB_RUN=d26 bash speedrun.sh`

if [ -z "$WANDB_RUN" ]; then
    # 默认使用 "dummy"：这是一个特殊情况，会跳过 wandb 日志记录
    WANDB_RUN=dummy
else
    # 如果设置了 WANDB_RUN 且不等于 dummy，则运行 online 模式
    if [ "$WANDB_RUN" != "dummy" ]; then
        # 检查是否提供了 API KEY
        if [ -z "$WANDB_API_KEY" ]; then
            echo "错误: 检测到 WANDB_RUN=$WANDB_RUN 为 Online 模式，但未检测到 WANDB_API_KEY"
            exit 1
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 日志系统重置
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器训练
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 等待预训练数据下载完成，然后进行预训练
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=18 $fp8_arg --target-param-data-ratio=8.25 --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE

# -----------------------------------------------------------------------------
# 指令微调数据集下载，并进行指令微调和评测
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ]; then
    echo "文件不存在，正在从 S3 下载..."
    curl -L -o "$IDENTITY_FILE" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
else
    echo "文件已存在，跳过下载: $IDENTITY_FILE"
fi
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# 命令行聊天测试
python -m scripts.chat_cli -p "Why is the sky blue?"

# web聊天测试
python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 生成报告
python -m nanochat.report generate
