# -*- coding: utf-8 -*-
import os
import json
import argparse
import time
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import APIs
import google.generativeai as genai
from openai import OpenAI


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
# 2. 建立 API 呼叫器 (Evaluator)
# ==========================================

class ModelEvaluator:
    def __init__(self, api_type="gemini", model_name="gemini-1.5-pro", api_key=None, base_url=None):
        self.api_type = api_type.lower()
        self.model_name = model_name
        self.base_url = base_url
        
        # 支援多個 API Keys，以逗號分隔
        self.api_keys = []
        if api_key:
            self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
            
        if not self.api_keys:
            # 依 API 類型嘗試讀取環境變數
            env_var = "OPENAI_API_KEY" if self.api_type == "openai" else "GEMINI_API_KEY"
            env_key = os.environ.get(env_var) or os.environ.get("GEMINI_API_KEY")
            if env_key:
                self.api_keys = [k.strip() for k in env_key.split(",") if k.strip()]
                
        if not self.api_keys:
            raise ValueError("No API keys found. Please set GEMINI_API_KEY or OPENAI_API_KEY environment variable.")
            
        import threading
        self.lock = threading.Lock()
        self.key_counter = 0
        
        print(f"[Evaluator] Initialized {self.api_type} client with {len(self.api_keys)} API key(s).")
        
        # 初始化客戶端
        if self.api_type == "gemini":
            # Gemini SDK 採用全域配置，多金鑰環境下呼叫時需加鎖並重新 configure
            pass
        elif self.api_type == "openai":
            # OpenAI 相容端點，為每個金鑰獨立建立客戶端，以支援完全執行緒安全的多金鑰平行呼叫
            self.clients = []
            actual_base_url = self.base_url
            is_google_key = any(k.startswith("AIza") or k.startswith("AQ.") for k in self.api_keys)
            if not actual_base_url and (is_google_key or "gemma" in self.model_name.lower()):
                actual_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            for key in self.api_keys:
                client = OpenAI(api_key=key, base_url=actual_base_url)
                self.clients.append(client)

    def get_client_or_key(self):
        """
        以 Round-Robin 方式輪詢分配 API 金鑰或客戶端。
        """
        with self.lock:
            idx = self.key_counter % len(self.api_keys)
            self.key_counter += 1
            
            if self.api_type == "openai":
                return self.clients[idx]
            else:
                return self.api_keys[idx]

    def evaluate_sample(self, system_prompt, user_query, target_seq, tools_desc, tool_steps, ref_ans):
        """
        對單個樣本進行評分呼叫，回傳解析後的 JSON 物件。
        """
        actual_model = self.model_name
        is_google_api = (self.api_type == "gemini" or 
                         (self.base_url and "generativelanguage.googleapis.com" in self.base_url) or
                         any(k.startswith("AIza") or k.startswith("AQ.") for k in self.api_keys))
        
        if is_google_api:
            cleaned = self.model_name.lower()
            if cleaned.startswith("google/"):
                cleaned = cleaned[7:]
            if not cleaned.startswith("models/"):
                cleaned = f"models/{cleaned}"
            actual_model = cleaned

        # 組裝輸入樣本資訊
        sample_info = {
            "user_utterance": user_query,
            "target_sequence": target_seq,
            "available_tools": tools_desc,
            "tool_steps": tool_steps,
            "reference_answer": ref_ans
        }
        
        user_message = f"Please evaluate the following synthetic data example:\n\n```json\n{json.dumps(sample_info, ensure_ascii=False, indent=2)}\n```"
        
        # 進行重試機制
        max_retries = 5
        backoff_sec = 2
        
        for attempt in range(max_retries):
            try:
                raw_response = ""
                client_or_key = self.get_client_or_key()
                
                if self.api_type == "gemini":
                    # 全域 SDK 配置與模型建立必須在加鎖的情況下進行以防止競爭
                    with self.lock:
                        genai.configure(api_key=client_or_key)
                        model = genai.GenerativeModel(
                            model_name=actual_model,
                            system_instruction=system_prompt
                        )
                    
                    response = model.generate_content(
                        user_message,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    raw_response = response.text
                elif self.api_type == "openai":
                    # 每個 Thread 使用獨立的 OpenAI 客戶端，完全平行處理，無競爭問題
                    response = client_or_key.chat.completions.create(
                        model=actual_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    raw_response = response.choices[0].message.content
                
                # 嘗試解析 JSON
                parsed_json = self._clean_and_parse_json(raw_response)
                return parsed_json, raw_response
                
            except Exception as e:
                print(f"[Warning] Attempt {attempt + 1} failed for example. Error: {str(e)}")
                if attempt < max_retries - 1:
                    error_msg = str(e)
                    is_rate_limit = ("429" in error_msg or 
                                     "quota" in error_msg.lower() or 
                                     "limit" in error_msg.lower() or 
                                     "exhausted" in error_msg.lower())
                    is_empty_response = ("expecting value" in error_msg.lower() and (not raw_response or not raw_response.strip()))
                    
                    if is_rate_limit:
                        import re
                        # 針對 Google AI Studio 429 限流進行智慧等待
                        sleep_time = 45
                        match = re.search(r"retry\s+in\s+([\d\.]+)", error_msg.lower())
                        if match:
                            try:
                                sleep_time = int(float(match.group(1))) + 2
                            except:
                                pass
                        else:
                            match_delay = re.search(r"retrydelay:\s*'(\d+)", error_msg.lower())
                            if match_delay:
                                try:
                                    sleep_time = int(match_delay.group(1)) + 2
                                except:
                                    pass
                        print(f"[RateLimit] Rate limit exceeded. Sleeping for {sleep_time} seconds before retry...")
                        time.sleep(sleep_time)
                    elif is_empty_response:
                        # 伺服器超載或暫時阻斷造成的空白回應，等待 15 秒以待恢復
                        sleep_time = 15
                        print(f"[EmptyResponse] Received empty response from model. Sleeping for {sleep_time} seconds before retry...")
                        time.sleep(sleep_time)
                    else:
                        time.sleep(backoff_sec)
                        backoff_sec *= 2
                else:
                    # 超過重試次數，返回錯誤資訊
                    return {
                        "error": f"Evaluation failed: {str(e)}",
                        "total_score": 0,
                        "overall_comment": f"Execution error: {str(e)}"
                    }, ""

    def _clean_and_parse_json(self, text):
        """
        清理 LLM 回傳內容並解析為 JSON。
        """
        if not text:
            return {}
        text = text.strip()
        
        # 去除 thinking block (如 <thought>...</thought>)
        if "<thought>" in text and "</thought>" in text:
            text = text.split("</thought>", 1)[-1].strip()
            
        # 去除 markdown 外殼 (如 ```json ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        return json.loads(text)


# ==========================================
# 3. 批次處理與平行化評分 (包含工具輸出壓縮)
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


def run_evaluation(args):
    # 1. 載入評分 System Prompt
    print(f"Loading prompt rules from: {args.prompt}")
    try:
        system_prompt = load_evaluation_prompt(args.prompt)
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
        # 預設嘗試以自定義文字格式解析
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
        
    # 4. 初始化評估器
    evaluator = ModelEvaluator(
        api_type=args.api_type,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 5. 多執行緒平行處理
    results = []
    
    # 輔助函式用於多執行緒包裝
    def process_row(idx, row):
        # 提取欄位
        rec_id = row.get("id") or f"idx_{idx}"
        source = row.get("source") or "unknown"
        target_seq = compact_target_sequence(clean_str_field(row.get("msg")))
        
        # 處理 struct/dict 欄位並轉為 JSON 序列化安全格式
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
        
        # 壓縮 tool_steps 中的工具呼叫與結果欄位，防止 Token 膨脹
        tool_steps_compacted = []
        for step in tool_steps:
            if isinstance(step, dict):
                compacted_step = step.copy()
                if "tool_call_line" in compacted_step:
                    compacted_step["tool_call_line"] = compact_tool_line(compacted_step["tool_call_line"])
                if "tool_result_line" in compacted_step:
                    compacted_step["tool_result_line"] = compact_tool_line(compacted_step["tool_result_line"])
                tool_steps_compacted.append(compacted_step)
            else:
                tool_steps_compacted.append(step)
        
        # 呼叫 API 評估
        evaluation, raw_text = evaluator.evaluate_sample(
            system_prompt=system_prompt,
            user_query=user_query,
            target_seq=target_seq,
            tools_desc=tools_desc,
            tool_steps=tool_steps_compacted,
            ref_ans=ref_ans
        )
        
        # 確保 evaluation 為字典型態，若為 list 則解包，避免屬性錯誤
        if isinstance(evaluation, list) and len(evaluation) > 0:
            evaluation = evaluation[0]
        if not isinstance(evaluation, dict):
            evaluation = {}
            
        # 回傳組裝資料 (以原始 row 所有欄位為基礎進行組裝)
        output_row = dict(row)
        
        # 加上舊程式相容之欄位 (僅供腳本內部使用)
        output_row["user_query"] = json.dumps(user_query, ensure_ascii=False) if isinstance(user_query, dict) else user_query
        output_row["target_sequence"] = target_seq
        
        # 將評估的分數與評論整合至同一個 'evaluation' 欄位
        output_row["evaluation"] = {
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
            # "raw_eval_response": raw_text,
            # "eval_parsed": json.dumps(evaluation, ensure_ascii=False)
        }
            
        return output_row


    print(f"Starting parallel evaluation with {args.concurrency} worker threads...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_row, i, row): i for i, row in df_eval.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[Error] Thread execution crashed: {e}")
                
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    
    # 6. 輸出結果與產生統計報表
    df_results = pd.DataFrame(results)
    
    output_ext = os.path.splitext(args.output)[-1].lower()
    if output_ext == ".parquet":
        df_results.to_parquet(args.output)
    else:
        # 預設輸出 JSONL
        df_results.to_json(args.output, orient="records", lines=True, force_ascii=False)
    
    print(f"Saved evaluation results to: {args.output}")
    
    # 7. 印出統計摘要
    print("\n" + "=" * 50)
    print(" EVALUATION STATISTICAL SUMMARY ")
    print("=" * 50)
    print(f"Total Evaluated Samples: {len(df_results)}")
    
    if len(df_results) > 0:
        # 從 evaluation 字典欄位提取評分
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
# 4. CLI 參數解析與進入點
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STITCH-S Synthetic Data Evaluator using Gemma4 / Gemini API")
    parser.add_argument(
        "--input", 
        default=DEFAULT_INPUT_PATH, 
        help="Path to the input parquet dataset."
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
        "--concurrency", 
        type=int, 
        default=5, 
        help="Number of parallel request threads."
    )
    parser.add_argument(
        "--api-type", 
        choices=["gemini", "openai"], 
        default="gemini", 
        help="API client type: 'gemini' (using google-generativeai SDK) or 'openai' (OpenAI-compatible client)."
    )
    parser.add_argument(
        "--model", 
        default="gemini-1.5-pro", 
        help="Model name (e.g., 'gemini-1.5-pro', 'google/gemma-4-26B-A4B-it')."
    )
    parser.add_argument(
        "--api-key", 
        default=None, 
        help="API Key (uses GEMINI_API_KEY/OPENAI_API_KEY environment variable if not specified)."
    )
    parser.add_argument(
        "--base-url", 
        default=None, 
        help="Base URL for OpenAI client (e.g. for custom endpoints or vLLM)."
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
    
    args = parser.parse_args()
    run_evaluation(args)
