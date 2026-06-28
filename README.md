# Agent-STITCH-S 合成資料評估工具 (eval_stitch-s)

此專案為 **Agent-STITCH-S** 合成對話資料的自動化評估工具。透過呼叫大語言模型（如 Gemini 或 OpenAI 相容端點上的 Gemma 等模型），依據嚴格的「真人客服與語音助理」標準，對模型的推理路徑（Reasoning chunks）與語音輸出（Spoken chunks）進行多維度評分。

---

## 📌 評估核心指標 (16 分制)

評估模型會依據 `prompt.txt` 中定義的 **8 項指標**進行評分，每項 0 至 2 分，總分共 16 分：

1. **Speech-first 合規**：第一個推理標記 `[SOPR]` 前，是否有安全且具體可播放的語音內容。
2. **工具等待安全性**：在工具尚未回傳結果前（Pending 狀態），是否只說安全等待話術而不預設或承諾未知結果。
3. **時間因果一致性**：工具結果是否只在回傳後才被後續的推理或語音所引用，杜絕「偷看未來資訊」的情況。
4. **逐步更新能力**：工具結果回來後，是否能自然且逐步地更新與修正回答。
5. **STITCH 標記與 chunk 品質**：`[SOPR]`、`[EOPR]`、`[EOR]` 標記使用是否正確，且推理與語音內容是否有適當分離。
6. **使用者空窗期與 timestamp 合理性**：使用者等待時間（audible silence gap）是否合理（小於 15 秒為佳），等待期間是否有短語音覆蓋。
7. **槽位完整性與 Grounding**：工具參數與最終回答是否皆能由上下文合理推導，嚴禁憑空捏造具體實體（如時間、地點、店名等）。
8. **真人客服自然性、口語化與 TTS 可朗讀性**：spoken chunk 會被 TTS 直接念出，是否口語自然、可一口氣念順、不含 markdown／符號／網址／程式碼／未口語化的數字時間；並在錯誤或空結果時能給出可行的下一步。

### 評分品質保證機制

`prompt.txt` 內建數項機制以提升評審一致性與資料品質訊號（皆維持 8 維 16 分制）：

* **單一歸口、避免重複扣分**：每個缺陷只在「最直接違反」的維度扣分，並有全域歸口節與各維度的歸口註記交叉約束（例如 TTS 只在第 8 項計分、grounding 只在第 7 項計分）。
* **N/A 與邊界規則**：無工具／無 pending 的乾淨樣本不被誤扣；相鄰等級難判定時「從嚴取低」，但各維度明文特例優先。
* **資料隔離與防注入**：所有輸入（含 `tool_result`、`reference_answer`）一律視為不可信資料，注入／操控文字本身視為品質瑕疵扣分。
* **證據導向評分**：扣分必須引用 `target_sequence` 原文，評語一律以正體中文撰寫。
* **雙語（zh-TW／English）**：口語化／TTS 與語言一致性規則皆語言中立。

> **總分一致性**：`total_score` 一律由 8 個分項分數重新加總覆寫（兩條 pipeline 行為一致），模型自報的總分不會污染統計；分項分數自動箝制為整數 0/1/2。

---

## 📂 檔案結構說明

* `evaluate.py`：API 版主評估程式（Gemini / OpenAI 相容端點）。支援多執行緒並行評估、自動文字截斷壓縮機制、智慧 Rate-Limit 避讓重試等功能。
* `evaluate_ray.py`：**本地 GPU 版**評估程式，使用 **Ray Data + vLLM** 做離線批次推論，可在多台 H100 節點上資料平行高吞吐執行。
* `prompt.txt`：定義評估的 System Prompt 指引與各維度詳細評分規範。
* `generation_spoken_style.md`：**生成端**風格規範（由 `prompt.txt` 反向萃取），附加到資料生成的 prompt，確保生成器與評審標準一致（spoken chunk 口語化／TTS、與 8 維對齊的檢查表）。
* `run_experiment.sh`：用以執行 API 版抽樣基準測試的 Shell 腳本範例。
* `run_evaluation_ray.sh`：**多節點 SLURM** 批次作業腳本，自動拉起跨節點 Ray 叢集並執行 `evaluate_ray.py`。

---

## ⚡ 多節點 GPU 評估（Ray + vLLM，H100 × 8 × N 節點）

`evaluate_ray.py` 以「**純資料平行**」方式擴展：模型（如 `gemma-4-26B-A4B-it`，26B MoE）可放進單張 80GB H100，因此每張 GPU 跑一個獨立的 vLLM replica，`--concurrency 0` 會自動依「整個叢集」的 GPU 總數推導 replica 數，加節點即線性加速。

```bash
# 1. 依需求調整節點數與 partition
#    run_evaluation_ray.sh 內：#SBATCH --nodes=N、--partition=<支援跨節點的佇列>
# 2. 送出作業
sbatch run_evaluation_ray.sh
```

腳本會：在 head 節點啟動 Ray head、在其餘節點啟動 worker 連回 head、等待註冊完成後執行評估，最後（透過 trap）關閉叢集。

**關鍵效率設計：**

* **自動擴展 replica**：`--concurrency 0` → `總GPU數 / (tensor_parallel_size × pipeline_parallel_size)`，無需手動改數字。
* **Prefix caching**：所有請求共用同一段 ~3K token 的 system prompt，`enable_prefix_caching=True` 命中率近 100%，是最大的吞吐來源。
* **無 shuffle 合併**：推論輸出（id + 回應）以 pandas left-merge 併回原始資料，避免 Ray 分散式 join，且保留推論失敗的列（不會悄悄遺失）。
* **依賴只在缺少時安裝**：避免每次作業重裝拖慢啟動（強制重裝：`export FORCE_DEPS=1`）。

主要可調參數（throughput / 記憶體權衡）：

| 參數 | 說明 |
| --- | --- |
| `--concurrency` | vLLM replica 數；`0` = 自動吃滿叢集 |
| `--tensor-parallel-size` | 單一 replica 跨幾張 GPU（模型放不進單卡才需 > 1） |
| `--pipeline-parallel-size` | 單一 replica 跨幾個節點（巨模型才需 > 1，會啟用 `distributed_executor_backend="ray"`） |
| `--max-num-seqs` / `--max-num-batched-tokens` | 每次 iteration 的並行序列數 / token 數，越大吞吐越高但更吃顯存 |
| `--kv-cache-dtype fp8` | KV cache 量化，省顯存、提吞吐（評分品質有極小影響，可改 `auto`） |
| `--max-tokens` | 輸出上限；評分 JSON 很短，調小可提升並行度 |

---

## 🚀 快速開始

### 1. 環境需求

請確保已安裝以下 Python 套件：

```bash
pip install pandas tqdm openai google-generativeai pyarrow fastparquet
```

### 2. 準備 API 金鑰

設定環境變數（推薦）或於執行時帶入參數：

* **Gemini API**：`export GEMINI_API_KEY="your_api_key"`
* **OpenAI API**：`export OPENAI_API_KEY="your_api_key"`

### 3. 執行評估

您可以直接使用 `run_experiment.sh` 執行抽樣測試：

```bash
bash run_experiment.sh
```

或者使用 `evaluate.py` 自訂更詳細的參數：

```bash
python evaluate.py \
    --input "path/to/dataset.parquet" \
    --output "eval_results.jsonl" \
    --model "gemini-1.5-pro" \
    --api-type gemini \
    --sample-size 50 \
    --concurrency 5
```
