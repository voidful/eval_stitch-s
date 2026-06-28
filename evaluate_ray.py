# -*- coding: utf-8 -*-
import os
import json
import argparse
import time
import re
import pandas as pd
import numpy as np
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_processor

# ==========================================
# 1. 讀取 Prompt 評分標準與設定輸出格式
# ==========================================

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")
DEFAULT_INPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stitch-s_train_with_tool.parquet")

JSON_OUTPUT_INSTRUCTION = """
---
請「務必」以 JSON 格式輸出評估結果。請勿包含任何 Markdown 標記（如 ```json）或額外的說明文字。

【硬性規則】
- 8 個分項分數（speech_first_score、tool_waiting_safety_score、temporal_causality_score、incremental_update_score、stitch_markup_score、silence_gap_score、grounding_score、naturalness_score）皆為必填，缺一不可；每個值只能是整數 0、1 或 2，不可為小數、字串或 null。
- timing_estimated 必為布林值 true／false（不可為字串）。
- 所有 *_comment 欄位皆必須提供（可簡短，但不可省略鍵），且一律以正體中文（臺灣）撰寫。
- total_score 必須嚴格等於 8 個分項分數之總和。
- 整個輸出必須是單一合法 JSON 物件，不得有任何 JSON 以外的文字、註解或 ``` 圍欄。

輸出的 JSON 結構必須完全符合以下欄位與格式：

{
  "speech_first_score": <0, 1, 2>,
  "speech_first_comment": "<關於 Speech-first 合規評語>",
  "tool_waiting_safety_score": <0, 1, 2>,
  "tool_waiting_safety_comment": "<關於 工具等待安全性評語>",
  "temporal_causality_score": <0, 1, 2>,
  "temporal_causality_comment": "<關於 時間因果一致性評語>",
  "incremental_update_score": <0, 1, 2>,
  "incremental_update_comment": "<關於 逐步更新能力評語>",
  "stitch_markup_score": <0, 1, 2>,
  "stitch_markup_comment": "<關於 STITCH 標記與 chunk 品質評語>",
  "silence_gap_score": <0, 1, 2>,
  "silence_gap_comment": "<關於 使用者空窗期與 timestamp 合理性評語>",
  "grounding_score": <0, 1, 2>,
  "grounding_comment": "<關於 使用者前提、槽位完整性與 tool argument grounding評語>",
  "naturalness_score": <0, 1, 2>,
  "naturalness_comment": "<關於 真人客服自然性、口語化與 TTS 可朗讀性評語>",
  "timing_estimated": <true/false>,
  "total_score": <必須嚴格等於上述 8 個分項分數之總和（0-16）；輸出前請自行加總核對>,
  "overall_comment": "<整體評核意見>"
}
"""

def load_evaluation_prompt(prompt_path=DEFAULT_PROMPT_PATH):
    """
    載入 prompt.txt 並附加強制的 JSON 格式說明。
    """
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Cannot find evaluation prompt file: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    
    # 結合原始評分標準與 JSON 輸出指示
    full_prompt = base_prompt.strip() + "\n" + JSON_OUTPUT_INSTRUCTION.strip()
    return full_prompt


# ==========================================
# 2. 工具輸出壓縮與清洗輔助函式
# ==========================================

def compact_text(text, max_chars=2400):
    """
    針對過長的文字進行截斷，保留頭部與尾部的關鍵資訊。
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    head = text[:1400]
    tail = text[-800:]
    return head + "\n...[TRUNCATED]...\n" + tail


def compact_tool_line(line_str, max_chars=2400):
    """
    對 <TOOL_CALL> 或 <TOOL_RESULT> 中的 JSON 內容進行值壓縮，以避免破壞 JSON 結構。
    """
    if not isinstance(line_str, str):
        return line_str
    
    tag_start = None
    tag_end = None
    inner_json_str = None
    
    if line_str.startswith("<TOOL_CALL>") and line_str.endswith("</TOOL_CALL>"):
        tag_start, tag_end = "<TOOL_CALL>", "</TOOL_CALL>"
        inner_json_str = line_str[len("<TOOL_CALL>"): -len("</TOOL_CALL>")].strip()
    elif line_str.startswith("<TOOL_RESULT>") and line_str.endswith("</TOOL_RESULT>"):
        tag_start, tag_end = "<TOOL_RESULT>", "</TOOL_RESULT>"
        inner_json_str = line_str[len("<TOOL_RESULT>"): -len("</TOOL_RESULT>")].strip()
        
    if inner_json_str:
        try:
            data = json.loads(inner_json_str)
            # 遞迴壓縮 dict/list 中的長字串值
            def recurse_compact(obj):
                if isinstance(obj, dict):
                    return {k: recurse_compact(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recurse_compact(x) for x in obj]
                elif isinstance(obj, str):
                    return compact_text(obj, max_chars=max_chars)
                else:
                    return obj
            
            compacted_data = recurse_compact(data)
            return f"{tag_start}{json.dumps(compacted_data, ensure_ascii=False)}{tag_end}"
        except:
            return f"{tag_start}{compact_text(inner_json_str, max_chars=max_chars)}{tag_end}"
            
    return compact_text(line_str, max_chars=max_chars)


def compact_target_sequence(target_seq_str):
    """
    對整個 trajectory (msg) 中的所有 <TOOL_CALL> 與 <TOOL_RESULT> 區塊進行壓縮替換。
    """
    if not isinstance(target_seq_str, str):
        return target_seq_str
        
    def call_replacer(match):
        return compact_tool_line(match.group(0))
        
    def result_replacer(match):
        return compact_tool_line(match.group(0))
        
    compacted = re.sub(r"<TOOL_CALL>.*?</TOOL_CALL>", call_replacer, target_seq_str, flags=re.DOTALL)
    compacted = re.sub(r"<TOOL_RESULT>.*?</TOOL_RESULT>", result_replacer, compacted, flags=re.DOTALL)
    
    return compacted


def clean_input_data(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return clean_input_data(parsed)
        except:
            return {}
    if isinstance(val, np.ndarray):
        val_list = val.tolist()
        if isinstance(val_list, list) and len(val_list) > 0:
            return clean_input_data(val_list[0])
        elif isinstance(val_list, dict):
            return val_list
        return {}
    if isinstance(val, list) and len(val) > 0:
        return clean_input_data(val[0])
    return {}


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj


def clean_str_field(val):
    """
    避免 Pandas NaN 轉為字串 "nan" 的輔助函式。
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def parse_txt_file(file_path):
    """
    解析自定義文字格式的資料集檔案。
    每一筆 record 包含：
      1. ID
      2. Source
      3. User query
      4. Msg (STITCH-S Trajectory)
      5. JSON Metadata (Input struct)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    records = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    
    while i < n:
        # 略過空白行
        if not lines[i].strip():
            i += 1
            continue
        
        # 預期 record 開始：ID、Source、User
        if i + 2 >= n:
            break
            
        record_id = lines[i].strip()
        source = lines[i+1].strip()
        user_query = lines[i+2].strip()
        
        i += 3
        
        # 蒐集 msg 行，直到遇到 JSON 開始標記 '{'
        msg_lines = []
        while i < n:
            line = lines[i]
            if line.strip() == "{":
                break
            msg_lines.append(line)
            i += 1
            
        msg = "\n".join(msg_lines).strip()
        
        # 解析 JSON 區塊
        json_lines = []
        brace_count = 0
        json_started = False
        
        while i < n:
            line = lines[i]
            json_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if "{" in line:
                json_started = True
            
            i += 1
            if json_started and brace_count == 0:
                break
                
        json_str = "\n".join(json_lines).strip()
        try:
            input_data = json.loads(json_str)
        except Exception as e:
            print(f"[Warning] Failed to parse JSON block for record {record_id}: {e}")
            input_data = {}
            
        records.append({
            "id": record_id,
            "source": source,
            "user": user_query,
            "msg": msg,
            "input": input_data
        })
        
    return pd.DataFrame(records)


# ==========================================
# 3. Ray Data 全域變數與 Batch Inference 映射函式
# ==========================================

SYSTEM_PROMPT = ""
NO_COMPACT = False
MAX_OUTPUT_TOKENS = 4096

def preprocess_eval(row):
    rec_id = row.get("id")
    target_seq_raw = clean_str_field(row.get("msg"))
    
    if NO_COMPACT:
        target_seq = target_seq_raw
    else:
        target_seq = compact_target_sequence(target_seq_raw)
        
    input_data = clean_input_data(row.get("input"))
    input_data = make_json_serializable(input_data)
    
    user_query_zh = clean_str_field(row.get("user"))
    user_query_en = clean_str_field(input_data.get("user"))
    
    if user_query_zh and user_query_en and user_query_zh != user_query_en:
        user_query = {
            "zh_TW": user_query_zh,
            "en": user_query_en
        }
    else:
        user_query = user_query_zh if user_query_zh else (user_query_en if user_query_en else "")
        
    tools_desc = make_json_serializable(input_data.get("available_tools") or [])
    tool_steps = make_json_serializable(input_data.get("tool_steps") or [])
    ref_ans = clean_str_field(input_data.get("reference_answer"))
    
    # 壓縮 tool_steps
    tool_steps_compacted = []
    for step in tool_steps:
        if isinstance(step, dict):
            compacted_step = step.copy()
            if not NO_COMPACT:
                if "tool_call_line" in compacted_step:
                    compacted_step["tool_call_line"] = compact_tool_line(compacted_step["tool_call_line"])
                if "tool_result_line" in compacted_step:
                    compacted_step["tool_result_line"] = compact_tool_line(compacted_step["tool_result_line"])
            tool_steps_compacted.append(compacted_step)
        else:
            tool_steps_compacted.append(step)
            
    sample_info = {
        "user_utterance": user_query,
        "target_sequence": target_seq,
        "available_tools": tools_desc,
        "tool_steps": tool_steps_compacted,
        "reference_answer": ref_ans
    }
    
    user_message = f"Please evaluate the following synthetic data example:\n\n```json\n{json.dumps(sample_info, ensure_ascii=False, indent=2)}\n```"
    
    return {
        "id": rec_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
    }


def postprocess_eval(row):
    return {
        "id": row["id"],
        "raw_response": row.get("generated_text") or row.get("response") or "",
    }


def clean_and_parse_json(text):
    if not text:
        return {}
    text = text.strip()
    
    if "<thought>" in text and "</thought>" in text:
        text = text.split("</thought>", 1)[-1].strip()
        
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```json") or lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    try:
        return json.loads(text)
    except Exception:
        return {}


SCORE_KEYS = [
    "speech_first_score", "tool_waiting_safety_score", "temporal_causality_score",
    "incremental_update_score", "stitch_markup_score", "silence_gap_score",
    "grounding_score", "naturalness_score",
]


def _coerce_score(value):
    """將分項分數安全轉為 0/1/2 整數；缺漏或無法解析時回傳 0。"""
    try:
        iv = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return min(2, max(0, iv))


def build_eval_struct(raw_response):
    """
    將單筆 LLM 原始回應字串解析為標準化的評分結構 (eval_struct)。
    為純函式，方便在 driver 端以 pandas apply 使用，避免額外的 Ray shuffle。
    """
    evaluation = clean_and_parse_json(raw_response or "")

    if isinstance(evaluation, list) and len(evaluation) > 0:
        evaluation = evaluation[0]
    if not isinstance(evaluation, dict):
        evaluation = {}

    eval_struct = {
        "speech_first_score": evaluation.get("speech_first_score", 0),
        "speech_first_comment": evaluation.get("speech_first_comment", ""),
        "tool_waiting_safety_score": evaluation.get("tool_waiting_safety_score", 0),
        "tool_waiting_safety_comment": evaluation.get("tool_waiting_safety_comment", ""),
        "temporal_causality_score": evaluation.get("temporal_causality_score", 0),
        "temporal_causality_comment": evaluation.get("temporal_causality_comment", ""),
        "incremental_update_score": evaluation.get("incremental_update_score", 0),
        "incremental_update_comment": evaluation.get("incremental_update_comment", ""),
        "stitch_markup_score": evaluation.get("stitch_markup_score", 0),
        "stitch_markup_comment": evaluation.get("stitch_markup_comment", ""),
        "silence_gap_score": evaluation.get("silence_gap_score", 0),
        "silence_gap_comment": evaluation.get("silence_gap_comment", ""),
        "grounding_score": evaluation.get("grounding_score", 0),
        "grounding_comment": evaluation.get("grounding_comment", ""),
        "naturalness_score": evaluation.get("naturalness_score", 0),
        "naturalness_comment": evaluation.get("naturalness_comment", ""),
        "timing_estimated": evaluation.get("timing_estimated", False),
        "total_score": evaluation.get("total_score", 0),
        "overall_comment": evaluation.get("overall_comment", ""),
    }
    
    # 強制分項為 0/1/2 整數、timing_estimated 為布林，並以 8 項之和覆寫 total_score
    # （缺鍵或型別錯誤時 _coerce_score 會回 0，避免靜默把整筆總分算錯）
    for k in SCORE_KEYS:
        eval_struct[k] = _coerce_score(eval_struct.get(k, 0))
    eval_struct["total_score"] = sum(eval_struct[k] for k in SCORE_KEYS)
    eval_struct["timing_estimated"] = bool(eval_struct.get("timing_estimated", False))

    return eval_struct


def wait_for_cluster(expected_gpus, timeout=600):
    """
    多節點啟動時，worker 節點向 head 註冊需要時間。
    在送出推論前等待叢集 GPU 數達到預期，避免只用到 head 節點的 GPU。
    回傳實際偵測到的 GPU 數。
    """
    have = int(ray.cluster_resources().get("GPU", 0))
    if expected_gpus <= 0 or have >= expected_gpus:
        return have

    waited = 0
    while waited < timeout:
        have = int(ray.cluster_resources().get("GPU", 0))
        if have >= expected_gpus:
            break
        print(f"[Cluster] Waiting for workers to join: {have}/{expected_gpus} GPUs ready ...")
        time.sleep(5)
        waited += 5
    return int(ray.cluster_resources().get("GPU", 0))


# ==========================================
# 4. 主評估流水線
# ==========================================

def run_evaluation(args):
    # 初始化全域變數
    global SYSTEM_PROMPT, NO_COMPACT, MAX_OUTPUT_TOKENS
    NO_COMPACT = args.no_compact
    MAX_OUTPUT_TOKENS = args.max_tokens

    # 1. 載入評分 System Prompt
    print(f"Loading prompt rules from: {args.prompt}")
    try:
        SYSTEM_PROMPT = load_evaluation_prompt(args.prompt)
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return
    
    # 2. 載入資料集
    print(f"Loading input dataset: {args.input}")
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    ext = os.path.splitext(args.input)[-1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(args.input)
    elif ext == ".jsonl":
        df = pd.read_json(args.input, orient="records", lines=True)
    elif ext == ".txt":
        df = parse_txt_file(args.input)
    else:
        try:
            df = parse_txt_file(args.input)
        except Exception as e:
            print(f"Failed parsing as text format: {e}. Trying pandas default read...")
            df = pd.read_parquet(args.input)
            
    print(f"Dataset successfully loaded. Total rows: {len(df)}")
    
    # 3. 抽樣處理
    if args.sample_size and args.sample_size < len(df):
        print(f"Sampling {args.sample_size} rows for evaluation (seed={args.seed})...")
        df_eval = df.sample(n=args.sample_size, random_state=args.seed).copy()
    else:
        df_eval = df.copy()
        print(f"Evaluating all {len(df_eval)} rows...")
        
    # 保證每筆資料有唯一 ID，以利於進行 join
    ids = []
    seen = set()
    for idx, row in df_eval.iterrows():
        val = str(row.get("id", ""))
        if not val or val in seen:
            val = f"row_{idx}"
        seen.add(val)
        ids.append(val)
    df_eval["id"] = ids

    # 4. 初始化 Ray
    #    優先使用環境變數 RAY_ADDRESS (由 SLURM 腳本匯出)，其次 "auto"。
    #    在多節點 SLURM 作業中若無法連上既有叢集，必須直接報錯，
    #    否則會悄悄退化成單機 (local) 執行，浪費其餘節點的 GPU。
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=ray_address)
    except Exception as e:
        if num_nodes > 1:
            raise RuntimeError(
                f"Multi-node SLURM job ({num_nodes} nodes) but failed to connect to the Ray "
                f"cluster at address='{ray_address}'. Aborting instead of silently falling back "
                f"to single-node mode. Original error: {e}"
            )
        print("Could not connect to an existing Ray cluster. Starting a local Ray instance...")
        ray.init()

    # 4b. 等待 worker 節點完成註冊，並依「整個叢集」的 GPU 數自動推導 replica 數
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) or args.gpus_per_node
    expected_gpus = num_nodes * gpus_per_node if num_nodes > 1 else 0
    cluster_gpus = wait_for_cluster(expected_gpus)

    gpus_per_replica = max(1, args.tensor_parallel_size * args.pipeline_parallel_size)
    if args.concurrency and args.concurrency > 0:
        num_replicas = args.concurrency
    else:
        # 純資料平行：每張 GPU 一個 replica (tp=pp=1)，自動吃滿整個叢集
        num_replicas = max(1, cluster_gpus // gpus_per_replica)

    print(
        f"[Cluster] nodes={num_nodes}, total GPUs={cluster_gpus}, "
        f"GPUs/replica={gpus_per_replica}, vLLM replicas={num_replicas}"
    )
    if cluster_gpus < num_replicas * gpus_per_replica:
        print(
            f"[Warning] Requested {num_replicas} replicas x {gpus_per_replica} GPU "
            f"= {num_replicas * gpus_per_replica} GPUs, but only {cluster_gpus} are available. "
            f"Ray will autoscale up to what fits; check your SLURM allocation."
        )

    # 5. 轉換為 Ray Dataset
    canonical_ds = ray.data.from_pandas(df_eval)

    # 6. 配置 vLLM Engine
    print("Configuring vLLM engine processor...")
    engine_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_model_len": args.max_model_len,
        "enable_chunked_prefill": True,
        # 所有請求共用同一段 system prompt (~3K tokens)，prefix caching 命中率近 100%，是最大的吞吐優化
        "enable_prefix_caching": True,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "trust_remote_code": True,
    }
    # 單一 replica 需跨節點 (pipeline parallel) 時，必須改用 ray 後端;
    # 單節點內的 tensor parallel 維持預設的 "mp" 即可，速度較佳。
    if args.pipeline_parallel_size > 1:
        engine_kwargs["distributed_executor_backend"] = "ray"

    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_kwargs=engine_kwargs,
        # (n, n) = 固定 n 個 replica，批次作業不需要 autoscaling 的暖機延遲
        concurrency=(num_replicas, num_replicas),
        batch_size=args.batch_size,
        should_continue_on_error=True,
        max_concurrent_batches=args.max_concurrent_batches,
        experimental={"max_tasks_in_flight_per_actor": 16},
    )

    processor = build_processor(
        config,
        preprocess=preprocess_eval,
        postprocess=postprocess_eval,
    )

    # 7. 執行推論評估
    print("Running vLLM engine inference for evaluation...")
    start_time = time.time()
    eval_ds = processor(canonical_ds)

    # 8. 合併結果：vLLM 輸出僅含 id + raw_response，量很小。
    #    以 pandas 在 driver 端 left-merge 回原始資料，避免 Ray 的分散式 shuffle (join)，
    #    並用 how="left" 保留推論失敗的列 (raw_response 為空)，不會像 inner join 一樣悄悄遺失資料。
    print("Collecting inference outputs and merging (no shuffle)...")
    eval_df = pd.DataFrame(eval_ds.take_all())
    if eval_df.empty:
        eval_df = pd.DataFrame(columns=["id", "raw_response"])
    if "id" not in eval_df.columns:
        eval_df["id"] = pd.Series(dtype="object")
    if "raw_response" not in eval_df.columns:
        eval_df["raw_response"] = ""
    eval_df = eval_df[["id", "raw_response"]].drop_duplicates(subset=["id"], keep="first")

    df_results = df_eval.merge(eval_df, on="id", how="left")
    missing = int(df_results["raw_response"].isna().sum())
    if missing:
        print(f"[Warning] {missing} row(s) produced no inference output (errors/timeouts). Scored as 0.")
    df_results["raw_response"] = df_results["raw_response"].fillna("")
    df_results["evaluation"] = df_results["raw_response"].apply(build_eval_struct)

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

    # 8. 儲存結果
    output_ext = os.path.splitext(args.output)[-1].lower()
    if output_ext == ".parquet":
        df_results.to_parquet(args.output)
    else:
        df_results.to_json(args.output, orient="records", lines=True, force_ascii=False)
    
    print(f"Saved evaluation results to: {args.output}")
    
    # 9. 印出統計摘要
    print("\n" + "=" * 50)
    print(" EVALUATION STATISTICAL SUMMARY (RAY+VLLM) ")
    print("=" * 50)
    print(f"Total Evaluated Samples: {len(df_results)}")
    
    if len(df_results) > 0:
        total_scores = df_results["evaluation"].apply(lambda x: x.get("total_score", 0) if isinstance(x, dict) else 0)
        avg_score = total_scores.mean()
        pass_rate = (total_scores >= args.pass_threshold).mean() * 100
        
        print(f"Average Total Score: {avg_score:.2f} / 16")
        print(f"Pass Rate (Score >= {args.pass_threshold}): {pass_rate:.1f}%")
        print("\nAverage scores by category (0-2):")
        categories = [
            ("Speech-first 合規", "speech_first_score"),
            ("工具等待安全性", "tool_waiting_safety_score"),
            ("時間因果一致性", "temporal_causality_score"),
            ("逐步更新能力", "incremental_update_score"),
            ("STITCH 標記與 chunk 品質", "stitch_markup_score"),
            ("使用者空窗期與 timestamp 合理性", "silence_gap_score"),
            ("使用者前提/槽位/grounding", "grounding_score"),
            ("真人客服自然性/口語化/TTS可朗讀", "naturalness_score")
        ]
        for name, col in categories:
            cat_mean = df_results["evaluation"].apply(lambda x: x.get(col, 0) if isinstance(x, dict) else 0).mean()
            print(f" - {name:<30}: {cat_mean:.2f}")
    print("=" * 50)


# ==========================================
# 5. CLI 參數解析與進入點
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STITCH-S Synthetic Data Evaluator using Ray Data + vLLM")
    parser.add_argument(
        "--input", 
        default=DEFAULT_INPUT_PATH, 
        help="Path to the input dataset (.parquet, .jsonl, or .txt)."
    )
    parser.add_argument(
        "--output", 
        default="eval_results.jsonl", 
        help="Path to save the output evaluation results (.jsonl or .parquet)."
    )
    parser.add_argument(
        "--prompt", 
        default=DEFAULT_PROMPT_PATH, 
        help="Path to the system evaluation criteria prompt file."
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=50, 
        help="Number of samples to evaluate (omit or set to 0 to evaluate all)."
    )
    parser.add_argument(
        "--pass-threshold", 
        type=int, 
        default=12, 
        help="Threshold score for a sample to be considered passing (max 16)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for sampling."
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Disable tool call/result compaction and pass the full raw text to the LLM."
    )
    
    # vLLM/Ray 專用加速與系統參數
    parser.add_argument(
        "--model", 
        default="google/gemma-4-26B-A4B-it", 
        help="Model name or HF model ID."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max model length for the vLLM engine."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens for the generator."
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (GPUs per replica, within a node). Keep at 1 unless the model "
             "does not fit on a single GPU."
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size (split one replica across nodes). Only needed for models that "
             "do not fit on a single node; uses distributed_executor_backend='ray'."
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="GPUs per node, used to compute expected cluster size when SLURM_GPUS_ON_NODE is unset."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization fraction for vLLM."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Number of concurrent vLLM replicas. 0 (default) = auto: total cluster GPUs / "
             "(tensor_parallel_size * pipeline_parallel_size), so it fills every node automatically."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="vLLM request batch size."
    )
    parser.add_argument(
        "--max-concurrent-batches",
        type=int,
        default=8,
        help="Maximum concurrent batches per actor."
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="fp8",
        help="KV cache data type (e.g. fp8, auto)."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration."
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=32768,
        help="Maximum number of batched tokens per iteration."
    )
    
    args = parser.parse_args()
    run_evaluation(args)
