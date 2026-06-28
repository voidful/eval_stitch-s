#!/bin/bash
#SBATCH --job-name=stitch_eval          # 工作名稱
#SBATCH --output=stitch_eval_%j.log     # 標準輸出日誌
#SBATCH --error=stitch_eval_%j.err      # 錯誤訊息日誌
#SBATCH --account=mst115022
#SBATCH --partition=8gpus               # ⚠ 多節點時請改成支援跨節點的 partition (8gpus 多為單節點佇列)
#SBATCH --nodes=2                       # 申請節點數 (多台機器)。單機測試改回 1
#SBATCH --ntasks-per-node=1             # 每節點 1 個任務，用來啟動該節點的 Ray daemon
#SBATCH --gres=gpu:8                    # 每節點 8 張 H100
#SBATCH --cpus-per-task=96              # 每節點 96 個 CPU 核心
#SBATCH --mem=1500G                     # 每節點 1500GB 系統記憶體
#SBATCH --time=02:00:00                 # 最長執行時間 2 小時

set -eo pipefail

# 無論成功或失敗都關閉 Ray 叢集 (fail-fast 時也能正確清理)
cleanup() {
    echo "Shutting down Ray cluster..."
    srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 ray stop || true
}
trap cleanup EXIT

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "=================================================="

# 載入模組與啟動 conda 環境
ml load miniconda3
ml load cuda/12.6
conda activate tp1

# 依賴只在缺少時才安裝，避免每次作業重裝拖慢啟動 / 破壞既有環境。
# 需要強制重裝時：在 sbatch 前 export FORCE_DEPS=1
if [ "${FORCE_DEPS:-0}" = "1" ] || ! python -c "import vllm, ray" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install vllm "ray[data]" "ray[default]" boto3 datasets
fi

# 停用實驗性的 vLLM V1 引擎以使用穩定的 V0 引擎，避免計算節點上因缺乏 JIT 編譯工具而初始化失敗。
# 若環境支援 (有 torch.compile 工具鏈)，改用 V1 (export VLLM_USE_V1=1) 通常吞吐更高。
export VLLM_USE_V1=0

# HuggingFace 模型快取：多節點時每個節點都要讀到權重。
# 請確保此路徑位於「共用儲存」(所有節點皆可存取)，模型才只需下載一次。
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

GPUS_PER_NODE=8

# ==========================================
# 啟動「多節點」Ray 叢集 (head + workers)
# ==========================================

# 1. 解析節點清單，第一個節點當 head
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# 取得 head 節點 IP (若回傳多個 IP，取第一個)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head        # evaluate_ray.py 會優先讀取此環境變數連上叢集
echo "Ray head node: $head_node ($ip_head)"

# 2. 啟動 Ray Head
echo "Starting Ray HEAD on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
        --node-ip-address="$head_node_ip" \
        --port=$port \
        --num-gpus="$GPUS_PER_NODE" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-96}" \
        --block &
sleep 20

# 3. 在其餘節點啟動 Ray Worker，連回 head
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
            --num-gpus="$GPUS_PER_NODE" \
            --num-cpus="${SLURM_CPUS_PER_TASK:-96}" \
            --block &
    sleep 5
done

# 等待所有 worker 完成註冊
sleep 15
echo "Ray cluster up. Resources:"
ray status || true

# ==========================================
# 執行核心 Python 評估程式碼 (在 head 節點上當作 driver)
# ==========================================
echo "Executing evaluate_ray.py..."

# 取得腳本所在目錄 (使腳本可於任意路徑執行)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# --concurrency 0 = 自動依「整個叢集」GPU 數推導 replica 數 (此例 2 節點 x 8 = 16 個 replica)
python3 "$DIR/evaluate_ray.py" \
    --input "$DIR/../eval/sampled_dataset.parquet" \
    --output "$DIR/eval_results_ray.jsonl" \
    --prompt "$DIR/prompt.txt" \
    --model "google/gemma-4-26B-A4B-it" \
    --sample-size 0 \
    --max-model-len 16384 \
    --max-tokens 1536 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpus-per-node $GPUS_PER_NODE \
    --gpu-memory-utilization 0.90 \
    --concurrency 0 \
    --batch-size 64 \
    --max-concurrent-batches 8 \
    --kv-cache-dtype "fp8" \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768

# 清理 (ray stop) 由 trap cleanup EXIT 自動執行
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
