#!/bin/bash
#SBATCH --job-name=stitch_eval          # 工作名稱
#SBATCH --output=stitch_eval_%j.log     # 標準輸出日誌
#SBATCH --error=stitch_eval_%j.err      # 錯誤訊息日誌
#SBATCH --account=mst115022
#SBATCH --partition=8gpus               # 使用 8gpus 佇列 (單節點快速排隊)
#SBATCH --nodes=1                       # 申請 1 個節點 (8張 H200 GPU)
#SBATCH --ntasks-per-node=1             # 每個節點起 1 個任務 (用來啟動 Ray node)
#SBATCH --gres=gpu:8                    # 每個節點跟系統申請 8 張 GPU (共 8 張 H200)
#SBATCH --cpus-per-task=96              # 每個節點配置 96 個 CPU 核心
#SBATCH --mem=1500G                     # 每個節點申請 1500GB 系統記憶體
#SBATCH --time=02:00:00                 # 最長執行時間限制 2 小時

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "=================================================="

# 載入模組與啟動 conda 環境
ml load miniconda3
ml load cuda/12.6
conda activate tp1

# 確保相依套件已安裝
pip install vllm "ray[data]" "ray[default]" boto3 datasets

# 停用實驗性的 vLLM V1 引擎以使用穩定的 V0 引擎，避免計算節點上因為缺乏 JIT 編譯工具而發生初始化失敗
export VLLM_USE_V1=0

# ==========================================
# 啟動單節點 Ray 叢集
# ==========================================

# 1. 取得 Head 節點的 IP 與 Hostname
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Ray Head Node IP: $ip_head"

# 2. 啟動 Ray Head 節點 (單節點包含 8 張 GPU)
echo "Starting Ray Head Node on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus=8 --block &
sleep 15

# 稍微等待節點啟動完畢
sleep 10
echo "Ray Cluster started successfully."

# ==========================================
# 執行核心 Python 評估程式碼
# ==========================================
echo "Executing evaluate_ray.py..."

# 取得腳本所在的目錄路徑 (使腳本可以在任何路徑被執行)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 "$DIR/evaluate_ray.py" \
    --input "$DIR/../eval/sampled_dataset.parquet" \
    --output "$DIR/eval_results_ray.jsonl" \
    --prompt "$DIR/prompt.txt" \
    --model "google/gemma-4-26B-A4B-it" \
    --max-model-len 16384 \
    --max-tokens 4096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --concurrency 8 \
    --batch-size 64 \
    --max-concurrent-batches 8 \
    --kv-cache-dtype "fp8" \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768

# ==========================================
# 清理與關閉 Ray 叢集
# ==========================================
echo "Shutting down Ray Cluster..."
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 ray stop

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
