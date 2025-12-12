# -*- coding: utf-8 -*-
# === RAG Benchmark: Ray Cluster + vLLM @Actor Version ===

from pathlib import Path
import gc
import os
import time
import asyncio
from statistics import mean
from typing import List
import itertools
import uuid

import ray
import torch
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# vLLM
from vllm.sampling_params import SamplingParams   # docs: https://docs.vllm.ai (mirrors)
from vllm.engine.arg_utils import AsyncEngineArgs  # AsyncEngineArgs dataclass
from vllm.engine.async_llm_engine import AsyncLLMEngine

# -------------------- 오프라인/캐시/공유 경로 --------------------
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

# -------------------- 유틸 --------------------
def load_vectorstore(faiss_dir: Path, embed_model: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)

def retrieve_context(vs: FAISS, query: str, k: int = TOP_K) -> str:
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content.strip() for d in docs)

# -------------------- vLLM Actor --------------------
@ray.remote(num_gpus=1, max_concurrency=1)
class VLLMWorker:
    def __init__(
        self,
        model_dir: str,
        dtype: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.60,
        kv_cache_dtype: str = "auto",      # "auto" | "fp8" | "fp8_e5m2" | "fp8_e4m3" (버전에 따라 미지원일 수 있음)
        quantization: str = None, 
        trust_remote_code: bool = True,
    ):
        import vllm
        from vllm.sampling_params import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        self.model_dir = model_dir
        self._sem = asyncio.Semaphore(1)
        self.vllm_version = getattr(vllm, "__version__", "unknown")
        print(f"[VLLMWorker] vLLM version: {self.vllm_version}")

        # 토크나이저는 HF로만 프롬프트 구성/토큰 카운트에 사용
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

        base_kwargs = dict(
            model=model_dir,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=1,      # Actor=1GPU
            tokenizer=self.tok.name_or_path,
            tokenizer_mode="auto",
        )

        # 버전 차이에 안전한 생성: 안 맞는 키워드는 제거하고 재시도
        trial_orders = [
            {**base_kwargs, "kv_cache_dtype": kv_cache_dtype, "quantization": quantization},
            {**base_kwargs, "kv_cache_dtype": kv_cache_dtype},   # quantization 제거
            {**base_kwargs}                                      # 모두 제거
        ]

        last_err = None
        for kw in trial_orders:
            try:
                self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kw))
                print(f"[VLLMWorker] Engine created with args keys: {list(kw.keys())}")
                break
            except TypeError as e:
                last_err = e
                print(f"[VLLMWorker] Retry without unsupported keys due to: {e}")
        else:
            raise last_err  # 모두 실패하면 예외

    async def generate_kpm(self, system_msg: str, user_msg: str, max_new_tokens: int = 1024):
        from vllm.sampling_params import SamplingParams

        async with self._sem:  # ★ 한 번에 하나만 추론
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            prompt_text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            eos_id = self.tok.eos_token_id
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_new_tokens,
                stop_token_ids=[eos_id] if eos_id is not None else None,  # HF와 종료 기준 정렬
            )

            request_id = str(uuid.uuid4())
            t_send = time.perf_counter()
            first_token_time = None
            final_output = None

            async for out in self.engine.generate(
                prompt=prompt_text,
                sampling_params=sampling,
                request_id=request_id
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                final_output = out

            t_done = time.perf_counter()

            # 토큰 수: vLLM가 반환한 실제 token_ids 우선 사용
            if final_output and final_output.outputs:
                try:
                    n_tokens = len(final_output.outputs[0].token_ids)
                except Exception:
                    text = final_output.outputs[0].text or ""
                    n_tokens = len(self.tok.encode(text, add_special_tokens=False))
            else:
                n_tokens = 0

            return {
                "ttft": (first_token_time - t_send) if first_token_time else None,
                "latency": (t_done - t_send),
                "n_tokens": n_tokens,
                "t_send": t_send,
                "t_first": first_token_time if first_token_time else t_done,
                "t_done": t_done,
            }



def main():
    # 고정 질문
    q1 = "상급자의 부당한 지시를 받은 경우 하급자가 취할 수 있는 절차를 단계별로 나열해줘. 그리고 etri 행동강령에 몇 장에 해당하는 내용이 나오는 지 알려줘."
    q2 = "사적 이해관계자'에 해당하는 사람은 총 몇 종류이며, 각각 누구를 의미하는지 정리해줘."
    sys_msg = (
        "너는 'ETRI 임직원 행동강령(etri.txt)'만을 근거로 간결하고 정확히 답한다. "
        "출처가 모호하면 '문서 근거 불충분'이라고 명시한다. 가능한 경우 장/절/조를 함께 표기한다."
    )

    print("Connecting to Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)
    print("Connected.")

    # RAG
    vs = load_vectorstore(FAISS_DIR, EMBED_MODEL)
    ctx_q1 = retrieve_context(vs, q1, k=TOP_K)
    ctx_q2 = retrieve_context(vs, q2, k=TOP_K)
    user_q1 = f"[컨텍스트]\n{ctx_q1}\n\n[질문]\n{q1}"
    user_q2 = f"[컨텍스트]\n{ctx_q2}\n\n[질문]\n{q2}"

    # 클러스터 GPU 수에 따라 액터 수 결정
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < 2:
        print(f"Warning: Only {available_gpus} GPU(s) available. Running with minimal actors.")
        n_q1_replicas = 1
        n_q2_replicas = 0
    else:
        n_q1_replicas = available_gpus // 2
        n_q2_replicas = available_gpus - n_q1_replicas

    print(f"Total GPUs in cluster: {available_gpus}")
    print(f"Creating {n_q1_replicas} actors for Q1 and {n_q2_replicas} actors for Q2...")

    # vLLM 튜닝(환경변수로 덮어쓰기 가능)
    vmax_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    vgpu_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.69"))
    vkv_dtype = os.getenv("VLLM_KV_CACHE_DTYPE", "auto")      # "auto" | "fp8" | "fp8_e5m2" | "fp8_e4m3"
    vquant = os.getenv("VLLM_QUANT", None)                    # ex) "awq", "gptq", "bitsandbytes"

    # Actor 생성
    workers_q1 = [
        VLLMWorker.remote(
            model_dir=str(Q1_PATH),
            max_model_len=vmax_len,
            gpu_memory_utilization=vgpu_util,
            kv_cache_dtype=vkv_dtype,
            quantization=vquant,
            dtype="auto",
            trust_remote_code=True,
        )
        for _ in range(n_q1_replicas)
    ]
    workers_q2 = [
        VLLMWorker.remote(
            model_dir=str(Q2_PATH),
            max_model_len=vmax_len,
            gpu_memory_utilization=vgpu_util,
            kv_cache_dtype=vkv_dtype,
            quantization=vquant,
            dtype="auto",
            trust_remote_code=True,
        )
        for _ in range(n_q2_replicas)
    ]

    # 라운드로빈 리스트 구성
    all_workers: List[ray.actor.ActorHandle] = []
    for a, b in itertools.zip_longest(workers_q1, workers_q2):
        if a: all_workers.append(a)
        if b: all_workers.append(b)

    if not all_workers:
        print("Error: No workers were created. Ensure GPUs are available in the Ray cluster.")
        return

    print(f"Total {len(all_workers)} actors created and ready.")

    # ---- 단일 라운드 실행 (동시 제출 n회) ----
    def run_round(n_concurrent: int):
        # -------------------- [PATCH] n=1에서 Q1+Q2를 같은 라운드에 함께 실행 --------------------
        if n_concurrent == 1:
            tasks = [("Q1", user_q1), ("Q2", user_q2)]
        else:
            tasks = [("Q1", user_q1) if i % 2 == 0 else ("Q2", user_q2) for i in range(n_concurrent)]
        expected_requests = len(tasks)
        # ------------------------------------------------------------------------------------------

        submit_times = {}     # {ObjectRef: driver_submit_time}
        completion_times = {} # {ObjectRef: driver_complete_time}
        futures = []

        for i, (_tag, user_msg) in enumerate(tasks):
            worker = all_workers[i % len(all_workers)]
            ref = worker.generate_kpm.remote(sys_msg, user_msg, MAX_NEW_TOKENS)
            submit_times[ref] = time.perf_counter()
            futures.append(ref)
        
        # 완료 시각(드라이버 시계) 기록 + 결과 수집
        results = []
        remaining = set(futures)
        
        while remaining:
            ready, remaining = ray.wait(list(remaining), num_returns=1, timeout=None)
            t_complete = time.perf_counter()
            for ref in ready:
                completion_times[ref] = t_complete
                results.append(ray.get(ref))  # 개별 회수
        
        assert len(results) == expected_requests  # [PATCH] 실제 요청 수 기준으로 검증
        
        # --- 집계 ---
        ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
        lats  = [r["latency"] for r in results]
        toks  = sum(r["n_tokens"] for r in results)
        ttft_avg = mean(ttfts) if ttfts else 0.0
        lat_avg  = mean(lats) if lats else 0.0
        # 드라이버 창(최초 submit ~ 최종 complete, 단일 시계)
        driver_window = (max(completion_times.values()) - min(submit_times.values()))
        driver_tput   = (toks / driver_window) if driver_window > 0 else 0.0
        # 유효 동시성(설명용): 총 서비스 시간 / 벽시계
        eff_parallelism = (sum(lats) / driver_window) if driver_window > 0 else float('nan')
        
        print(f"\n[ROUND {n_concurrent}]")
        print(f"Requests_in_round                 : {expected_requests}")  # [PATCH] 실제 요청 수 출력
        print(f"TTFT_avg_sec                      : {ttft_avg:.6f}")
        print(f"Latency_avg_sec                   : {lat_avg:.6f}")
        print(f"Latency_total_sec (driver window) : {driver_window:.6f}")
        print(f"Total_Throughput_tokens_per_sec   : {driver_tput:.6f}")
        print(f"Eff_parallelism_x                 : {eff_parallelism:.2f}")

    # ---- 라운드 실행 ----
    for n in ROUNDS:
        run_round(n)
        print(f"\n--- Flushing memory after round {n} ---")
        gc.collect()
        print("--- Memory flushed ---")

    ray.shutdown()
    print("\nBenchmark finished and Ray has been shut down.")

if __name__ == "__main__":
    main()
