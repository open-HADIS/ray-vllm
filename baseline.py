# Libraries
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import os  # added
import time  # added
import threading  # added
from statistics import mean  # added

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer  # added

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------- 오프라인/캐시/공유 경로 고정 (기존과 동일) --------------------
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/shared/hf-home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

FAISS_DIR   = Path("/mnt/shared/faiss_index")
EMBED_MODEL = "/mnt/shared/models/ko-sroberta-multitask"
Q1_PATH     = Path("/mnt/shared/models/qwen25_3b")
Q2_PATH     = Path("/mnt/shared/models/qwen25_1_5b_instruct")

TOP_K = 4
MAX_NEW_TOKENS = 1024
ROUNDS = [1, 6, 12, 24, 48]


# Vector Store loading (vs)
def load_vectorstore(faiss_dir: Path, embed_model: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)

# Context Top k 만큼 가져오기
def retrieve_context(vs: FAISS, query: str, k: int = TOP_K) -> str:
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content.strip() for d in docs)

# Local LLM
def load_local_llm(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        dtype="auto",
    )
    return tok, model

# 스트리밍 기반 생성: TTFT/Latency/토큰수 측정
@torch.inference_mode()
def chat_generate_kpm(tok, model, system_msg: str, user_msg: str, max_new_tokens: int = MAX_NEW_TOKENS):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    input_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)  # added

    t_send = time.perf_counter()  # added

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    first_token_time = None  # added
    chunks = []  # added
    for piece in streamer:
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now  # added
        chunks.append(piece)

    thread.join()
    t_done = time.perf_counter()  # added

    text = "".join(chunks).strip()  # added
    out_tokens = tok.encode(text, add_special_tokens=False)  # added
    return {
        "ttft": first_token_time - t_send if first_token_time else None,  # added
        "latency": t_done - t_send,                            # added
        "n_tokens": len(out_tokens),                           # added
        "t_first": first_token_time if first_token_time else t_done,  # added
        "t_done": t_done,                                      # added
        "t_send": t_send,                                      # added
    }

def main():
    # 질문 고정
    q1 = "상급자의 부당한 지시를 받은 경우 하급자가 취할 수 있는 절차를 단계별로 나열해줘. 그리고 etri 행동강령에 몇 장에 해당하는 내용이 나오는 지 알려줘."
    q2 = "사적 이해관계자'에 해당하는 사람은 총 몇 종류이며, 각각 누구를 의미하는지 정리해줘."

    sys_msg = (
        "너는 'ETRI 임직원 행동강령(etri.txt)'만을 근거로 간결하고 정확히 답한다. "
        "출처가 모호하면 '문서 근거 불충분'이라고 명시한다. 가능한 경우 장/절/조를 함께 표기한다."
    )

    # RAG
    vs = load_vectorstore(FAISS_DIR, EMBED_MODEL)
    ctx_q1 = retrieve_context(vs, q1, k=TOP_K)
    ctx_q2 = retrieve_context(vs, q2, k=TOP_K)

    user_q1 = f"[컨텍스트]\n{ctx_q1}\n\n[질문]\n{q1}"
    user_q2 = f"[컨텍스트]\n{ctx_q2}\n\n[질문]\n{q2}"

    # 모델 로드
    tok1, model1 = load_local_llm(Q1_PATH)
    tok2, model2 = load_local_llm(Q2_PATH)

    # 동시 접근 시 OOM 방지용 모델별 락 (한 모델당 1개 generate 동시 진행)  # added
    lock_q1 = threading.Lock()  # added
    lock_q2 = threading.Lock()  # added

    def _one_call(which: str):  # added
        if which == "Q1":
            with lock_q1:
                return chat_generate_kpm(tok1, model1, sys_msg, user_q1)
        else:
            with lock_q2:
                return chat_generate_kpm(tok2, model2, sys_msg, user_q2)

    # ---- 단일 라운드 실행 (동시 제출 n회) ----  # added
    def run_round(n_concurrent: int):
        # 태스크를 번갈아가며 생성(Q1/Q2 번갈이)  # added
        tags = [("Q1" if i % 2 == 0 else "Q2") for i in range(n_concurrent)]  # added

        t_round_start = time.perf_counter()  # added
        results = []  # added

        with ThreadPoolExecutor(max_workers=n_concurrent) as ex:
            futs = [ex.submit(_one_call, tag) for tag in tags]  # added
            for fut in as_completed(futs):
                results.append(fut.result())  # added

        t_round_end = time.perf_counter()  # added

        # KPM 집계  # added
        ttfts = [r["ttft"] for r in results if r["ttft"] is not None]  # added
        lats = [r["latency"] for r in results]  # added
        earliest_first = min(r["t_first"] for r in results)  # added
        latest_done = max(r["t_done"] for r in results)  # added
        window = max(1e-9, latest_done - earliest_first)  # added
        total_tokens = sum(r["n_tokens"] for r in results)  # added

        # 1) TTFT: per-request 평균  # added
        ttft_avg = mean(ttfts) if ttfts else 0.0  # added
        # 2) Latency(per-request 평균)  # added
        lat_avg = mean(lats) if lats else 0.0  # added
        # 3) Latency(total): 라운드 전체 벽시계  # added
        latency_total = t_round_end - t_round_start  # added
        # 4) Total_Throughput  # added
        total_throughput = total_tokens / window  # added

        # ---- 최종 출력: 4가지 KPM만 (라운드 헤더 포함) ----  # added
        print(f"[ROUND {n_concurrent}]")  # added
        print(f"TTFT_avg_sec: {ttft_avg:.6f}")  # added
        print(f"Latency_avg_sec: {lat_avg:.6f}")  # added
        print(f"Latency_total_sec: {latency_total:.6f}")  # added
        print(f"Total_Throughput_tokens_per_sec: {total_throughput:.6f}")  # added
        # 개행으로 라운드 시각적으로 구분  # added
        print()  # added

    # ---- 라운드 실행: [1, 6, 20, 40] ----
    for n in [1, 6, 12, 24, 48]:
        run_round(n)
        
        # [MODIFICATION] Clear memory after each round
        print(f"--- Flushing memory after round {n} ---")
        gc.collect()
        torch.cuda.empty_cache()
        print("--- Memory flushed ---\n")

if __name__ == "__main__":
    main()