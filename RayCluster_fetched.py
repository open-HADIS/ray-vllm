# -*- coding: utf-8 -*-
# === RAG Benchmark: Ray Cluster @Actor Version ===

# Libraries
from pathlib import Path
import gc
import os
import time
import threading
from statistics import mean
from typing import Dict, List
import itertools

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ray # [MODIFICATION] Import Ray

# -------------------- 오프라인/캐시/공유 경로 고정 (Baseline과 동일) --------------------
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


# -------------------- 유틸리티 함수 (Baseline과 동일) --------------------
def load_vectorstore(faiss_dir: Path, embed_model: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)

def retrieve_context(vs: FAISS, query: str, k: int = TOP_K) -> str:
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content.strip() for d in docs)

def load_local_llm(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        torch_dtype="auto",
    )
    return tok, model

# -------------------- [MODIFICATION] Ray Actor로 모델 래핑 --------------------
@ray.remote(num_gpus=1)
class ModelWorker:
    """
    Ray Actor that loads a model onto a dedicated GPU and handles inference requests.
    This replaces the manual locking mechanism from the baseline script.
    """
    def __init__(self, model_dir: str):
        # Load the model and tokenizer within the actor's process
        self.tok, self.model = load_local_llm(Path(model_dir))

    @torch.inference_mode()
    def generate_kpm(self, system_msg: str, user_msg: str, max_new_tokens: int = MAX_NEW_TOKENS):
        # This is the exact same logic from the baseline's chat_generate_kpm function
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        input_ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        t_send = time.perf_counter()

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        first_token_time = None
        chunks = []
        for piece in streamer:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            chunks.append(piece)

        thread.join()
        t_done = time.perf_counter()

        text = "".join(chunks).strip()
        out_tokens = self.tok.encode(text, add_special_tokens=False)
        
        # Return the same detailed dictionary required for KPM calculation
        return {
            "ttft": first_token_time - t_send if first_token_time else None,
            "latency": t_done - t_send,
            "n_tokens": len(out_tokens),
            "t_first": first_token_time if first_token_time else t_done,
            "t_done": t_done,
            "t_send": t_send,
        }

def main():
    # 질문 고정 (Baseline과 동일)
    q1 = "상급자의 부당한 지시를 받은 경우 하급자가 취할 수 있는 절차를 단계별로 나열해줘. 그리고 etri 행동강령에 몇 장에 해당하는 내용이 나오는 지 알려줘."
    q2 = "사적 이해관계자'에 해당하는 사람은 총 몇 종류이며, 각각 누구를 의미하는지 정리해줘."
    sys_msg = (
        "너는 'ETRI 임직원 행동강령(etri.txt)'만을 근거로 간결하고 정확히 답한다. "
        "출처가 모호하면 '문서 근거 불충분'이라고 명시한다. 가능한 경우 장/절/조를 함께 표기한다."
    )

    # [MODIFICATION] Connect to the Ray Cluster
    print("Connecting to Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)
    print("Connected.")

    # RAG (Baseline과 동일)
    vs = load_vectorstore(FAISS_DIR, EMBED_MODEL)
    ctx_q1 = retrieve_context(vs, q1, k=TOP_K)
    ctx_q2 = retrieve_context(vs, q2, k=TOP_K)
    user_q1 = f"[컨텍스트]\n{ctx_q1}\n\n[질문]\n{q1}"
    user_q2 = f"[컨텍스트]\n{ctx_q2}\n\n[질문]\n{q2}"

    # [MODIFICATION] 모델을 직접 로드하는 대신, Actor를 생성합니다.
    # 클러스터의 전체 GPU 수를 기반으로 Actor 수를 결정합니다.
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < 2:
        print(f"Warning: Only {available_gpus} GPU(s) available. Running with minimal actors.")
        n_q1_replicas = 1
        n_q2_replicas = 0
    else:
        n_q1_replicas = available_gpus // 2
        n_q2_replicas = available_gpus - n_q1_replicas
    
    print(f"Total GPUs in cluster: {available_gpus}")
    print(f"Creating {n_q1_replicas} actors for Model Q1 and {n_q2_replicas} for Model Q2...")

    workers_q1 = [ModelWorker.remote(str(Q1_PATH)) for _ in range(n_q1_replicas)]
    workers_q2 = [ModelWorker.remote(str(Q2_PATH)) for _ in range(n_q2_replicas)]
    
    # 워커들을 번갈아가며 사용하기 위해 리스트를 섞습니다.
    all_workers: List[ray.actor.ActorHandle] = list(itertools.chain.from_iterable(zip(workers_q1, workers_q2)))
    # GPU 수가 다를 경우 남은 워커 추가
    if len(workers_q1) > len(workers_q2):
        all_workers.extend(workers_q1[len(workers_q2):])
    elif len(workers_q2) > len(workers_q1):
        all_workers.extend(workers_q2[len(workers_q1):])

    if not all_workers:
        print("Error: No workers were created. Ensure GPUs are available in the Ray cluster.")
        return
        
    print(f"Total {len(all_workers)} actors created and ready.")

    # ---- 단일 라운드 실행 (동시 제출 n회) ----
    def run_round(n_concurrent: int):
        # [MOD] n=1이면 같은 라운드에서 Q1+Q2를 함께 실행
        if n_concurrent == 1:
            tasks = [("Q1", user_q1), ("Q2", user_q2)]
        else:
            # 기존 동작: Q1/Q2 번갈아 n_concurrent개 생성
            tasks = [("Q1", user_q1) if i % 2 == 0 else ("Q2", user_q2) for i in range(n_concurrent)]
        
        t_round_start = time.perf_counter()
        
        futures = []
        for i, (tag, user_msg) in enumerate(tasks):
            # Round-robin assignment of tasks to workers
            worker = all_workers[i % len(all_workers)]
            future = worker.generate_kpm.remote(sys_msg, user_msg)
            futures.append(future)
        
        # 모든 원격 태스크가 완료될 때까지 기다리고 결과를 가져옵니다.
        results = ray.get(futures)
        
        # [옵션] 각 요청별 결과 확인용
        print(f"\n--- Individual results for Round {n_concurrent} ---")
        for i, r in enumerate(results):
            print(f"  Request {i+1}: n_tokens={r.get('n_tokens', 0)}")

        t_round_end = time.perf_counter()

        # KPM 집계 (Baseline과 동일 로직)
        ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
        lats = [r["latency"] for r in results]
        total_tokens = sum(r["n_tokens"] for r in results)

        ttft_avg = mean(ttfts) if ttfts else 0.0
        lat_avg = mean(lats) if lats else 0.0
        latency_total = t_round_end - t_round_start
        
        # 분산 환경에서는 드라이버에서 측정한 latency_total로 처리량 계산
        total_throughput = total_tokens / latency_total if latency_total > 0 else 0.0

        # ---- 최종 출력 ----
        print(f"\n[ROUND {n_concurrent}]")
        print(f"Requests_in_round: {len(results)}")  # [MOD] 실제 요청 수 출력
        print(f"TTFT_avg_sec: {ttft_avg:.6f}")
        print(f"Latency_avg_sec: {lat_avg:.6f}")
        print(f"Latency_total_sec: {latency_total:.6f}")
        print(f"Total_Throughput_tokens_per_sec: {total_throughput:.6f}")

    # ---- 라운드 실행 ----
    for n in ROUNDS:
        run_round(n)
        # Actor 기반에서는 드라이버에서 메모리 플러싱의 효과가 적지만, 좋은 습관으로 유지
        print(f"\n--- Flushing memory after round {n} ---")
        gc.collect()
        print("--- Memory flushed ---")

    ray.shutdown()
    print("\nBenchmark finished and Ray has been shut down.")

if __name__ == "__main__":
    main()
