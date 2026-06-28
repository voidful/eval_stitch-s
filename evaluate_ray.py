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
請「務必」以 JSON 格式輸出評估結果。請勿包含任何 Markdown 標記（如 ```json）或額外的說明文字。輸出的 JSON 結構必須完全符合以下欄位與格式：

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
  "naturalness_comment": "<關於 真人客服自然性與任務可用性評語>",
  "timing_estimated": <true/false>,
  "total_score": <總分，範圍在 0-16 之間，應為上述 8 個項目分數之總和>,
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


def process_eval_row(row):
    raw_response = row.get("raw_response") or ""
    evaluation = clean_and_parse_json(raw_response)
    
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
    
    # 確保總分與分項分數相加一致
    score_keys = [
        "speech_first_score",
        "tool_waiting_safety_score",
        "temporal_causality_score",
        "incremental_update_score",
        "stitch_markup_score",
        "silence_gap_score",
        "grounding_score",
        "naturalness_score"
    ]
    inferred_total = sum(int(eval_struct.get(k, 0)) for k in score_keys)
    if eval_struct["total_score"] == 0 or eval_struct["total_score"] != inferred_total:
        eval_struct["total_score"] = inferred_total
        
    output_row = dict(row)
    output_row["evaluation"] = eval_struct
    return output_row


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
    try:
        ray.init(address="auto")
    except Exception:
        print("Could not connect to existing Ray cluster. Starting a local Ray instance...")
        ray.init()

    # 5. 轉換為 Ray Dataset
    canonical_ds = ray.data.from_pandas(df_eval)

    # 6. 配置 vLLM Engine
    print("Configuring vLLM engine processor...")
    config = vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_kwargs={
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "trust_remote_code": True,
        },
        concurrency=(args.concurrency, args.concurrency),
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
    
    # Join 回 canonical_ds 以合併欄位
    print("Joining results and parsing evaluation JSON...")
    joined_ds = canonical_ds.join(eval_ds, join_type="inner", on=["id"])
    processed_ds = joined_ds.map(process_eval_row)
    
    # 讀取全部結果回記憶體中的 pandas DataFrame
    results = processed_ds.take_all()
    df_results = pd.DataFrame(results)
    
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
            ("槽位與 tool argument grounding", "grounding_score"),
            ("真人客服自然性與任務可用性", "naturalness_score")
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
        help="Tensor parallel size for vLLM."
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
        default=8,
        help="Number of concurrent vLLM actors (engine instances)."
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
