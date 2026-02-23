from intent_detection.id import intent_detection
from llama_cpp import Llama
from typing import Any, List, Dict
import json
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from datetime import datetime


disable_progress_bars()

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
QA_PATH = "/Users/silei/Code/local_code/intent_detection/QA.json"
OUT_PATH = f"/Users/silei/Code/local_code/intent_detection/results_{RUN_TS}.json"


""""
TODO: Stop printing these logs

import os
from contextlib import redirect_stdout, redirect_stderr

with open(os.devnull, "w") as devnull:
    with redirect_stdout(devnull), redirect_stderr(devnull):
        llm_instruct = Llama.from_pretrained(
            repo_id="matrixportalx/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            verbose=False,
        )
"""


def main():

    llm_instruct = Llama.from_pretrained(
        repo_id="matrixportalx/Qwen2.5-7B-Instruct-GGUF",
	    filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        verbose=False,
    )
    logs = []
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_items: List[Dict[str, Any]] = json.load(f)

    for i, item in enumerate(qa_items, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        answer = intent_detection(question, llm_instruct)
        logs.append({
            "idx": i,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": answer,
            "result": True if ground_truth == answer else False
        })
        print(f"[{i}/{len(qa_items)}] answered")

    output = {
    "model": "matrixportalx/Qwen2.5-7B-Instruct-GGUF",
    "count": len(logs),
    "records": logs
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
    #print("OK")