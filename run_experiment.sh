#!/bin/bash


API_KEY=""
MODEL="google/gemma-4-26B-A4B-it"
SAMPLE_SIZE=10 #抽樣數量
SEED=42
CONCURRENCY=4

# 取得腳本所在的目錄路徑 (使腳本可以在任何路徑被執行)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================================="
echo "正在執行 10 筆不壓縮評估 (Uncompacted) 基準測試..."
conda run --no-capture-output -n env_stitch-s python "$DIR/evaluate.py" \
    --input "$DIR/sampled_dataset.parquet" \  #小量測試
    --output "$DIR/eval_results_no_compact.jsonl" \
    --sample-size $SAMPLE_SIZE \
    --seed $SEED \
    # --no-compact \  #是否擷取
    --api-type openai \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --concurrency $CONCURRENCY

echo "=========================================================="
echo "基準測試完成！"
