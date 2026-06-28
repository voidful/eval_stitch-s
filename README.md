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
8. **真人客服自然性與任務可用性**：語音輸出是否簡短自然，在發生錯誤或空結果時能否給出可行的下一步。

---

## 📂 檔案結構說明

* `evaluate.py`：主評估程式。支援多執行緒並行評估、自動文字截斷壓縮機制、智慧 Rate-Limit 避讓重試等功能。
* `prompt.txt`：定義評估的 System Prompt 指引與各維度詳細評分規範。
* `run_experiment.sh`：用以執行基準測試的 Shell 腳本範例。

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
