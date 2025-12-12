# -*- coding: utf-8 -*-
# === Serve-only RAG Benchmark: Ray Serve Deployments vs Rounds (version-agnostic) ===

from pathlib import Path
import gc, os, time, threading
from statistics import mean

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ray
from ray import serve

# -------------------- Config --------------------
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

# -------------------- Utils --------------------
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
        dtype="auto",
    )
    return tok, model

# -------------------- Serve Deployment --------------------
@serve.deployment  # 동시성은 내부 세마포어로 강제(버전 무관)
class ServeModelWorker:
    def __init__(self, model_dir: str):
        self.tok, self.model = load_local_llm(Path(model_dir))
        # 레플리카당 동시 1요청 강제 (버전별 옵션 차이 회피)
        self._sem = threading.BoundedSemaphore(value=1)

    @torch.inference_mode()
    def generate_kpm(self, system_msg: str, user_msg: str, max_new_tokens: int = MAX_NEW_TOKENS):
        with self._sem:  # 동시처리=1 보장
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            input_ids = self.tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            t_send = time.perf_counter()

            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.eos_token_id,
                streamer=streamer,
            )
            th = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            th.start()

            first_token_time = None
            chunks = []
            for piece in streamer:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(piece)

            th.join()
            t_done = time.perf_counter()

            text = "".join(chunks).strip()
            out_tokens = self.tok.encode(text, add_special_tokens=False)

            return {
                "ttft": first_token_time - t_send if first_token_time else t_done - t_send,
                "latency": t_done - t_send,
                "n_tokens": len(out_tokens),
            }

def run_serve_benchmark(user_q1: str, user_q2: str, sys_msg: str):
    print("\n" + "="*80)
    print("               🚀 STARTING BENCHMARK: Ray Serve (ONLY) 🚀")
    print("="*80 + "\n")

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < 1:
        print("Error: No GPUs available.")
        return

    n_q1 = max(1, available_gpus // 2)
    n_q2 = max(0, available_gpus - n_q1)
    print(f"Total GPUs: {available_gpus}. Deploying {n_q1} replicas for Q1 and {n_q2} for Q2...")

    # Q1 앱
    q1_app = ServeModelWorker.options(
        name="q1_worker",
        num_replicas=n_q1,
        ray_actor_options={"num_gpus": 1},
    ).bind(model_dir=str(Q1_PATH))
    serve.run(q1_app, name="q1_app", route_prefix=None)

    # Q2 앱(레플리카 0이면 생략)
    if n_q2 > 0:
        q2_app = ServeModelWorker.options(
            name="q2_worker",
            num_replicas=n_q2,
            ray_actor_options={"num_gpus": 1},
        ).bind(model_dir=str(Q2_PATH))
        serve.run(q2_app, name="q2_app", route_prefix=None)

    # 핸들 취득
    q1h = serve.get_deployment_handle("q1_worker", app_name="q1_app")
    q2h = serve.get_deployment_handle("q2_worker", app_name="q2_app") if n_q2 > 0 else None
    print("Serve handles are ready.")

    for n in ROUNDS:
        # -------------------- [PATCH] n=1에서 Q1+Q2를 같은 라운드에 함께 실행 --------------------
        if n == 1:
            tasks = [("Q1", user_q1), ("Q2", user_q2)]
        else:
            tasks = [("Q1", user_q1) if i % 2 == 0 else ("Q2", user_q2) for i in range(n)]
        # ----------------------------------------------------------------------------------------

        t0 = time.perf_counter()
        futures = []
        for tag, um in tasks:
            # Q2 레플리카가 없으면 안전하게 Q1로 라우팅(원래 코드 유지)
            h = q1h if (tag == "Q1" or q2h is None) else q2h
            futures.append(h.generate_kpm.remote(sys_msg, um))
        results = [r.result() for r in futures]
        t1 = time.perf_counter()

        ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
        lats  = [r["latency"] for r in results]
        toks  = sum(r["n_tokens"] for r in results)

        ttft_avg = mean(ttfts) if ttfts else 0.0
        lat_avg  = mean(lats) if lats else 0.0
        wall     = t1 - t0
        tput     = toks / wall if wall > 0 else 0.0

        print(f"\n[ROUND {n}]")
        print(f"  - Requests_in_round              : {len(results)}")  # 요청 수 명시
        print(f"  - TTFT_avg_sec                   : {ttft_avg:.6f}")
        print(f"  - Latency_avg_sec (per request)  : {lat_avg:.6f}")
        print(f"  - Latency_total_sec (wall time)  : {wall:.6f}")
        print(f"  - Total_Throughput_tokens_per_sec: {tput:.6f}")

    serve.shutdown()
    gc.collect()
    print("\n✅ Serve benchmark finished and Serve has been shut down.")

# -------------------- Main --------------------
def main():
    q1 = "상급자의 부당한 지시를 받은 경우 하급자가 취할 수 있는 절차를 단계별로 나열해줘. 그리고 etri 행동강령에 몇 장에 해당하는 내용이 나오는 지 알려줘."
    q2 = "사적 이해관계자'에 해당하는 사람은 총 몇 종류이며, 각각 누구를 의미하는지 정리해줘."
    sys_msg = (
        "너는 'ETRI 임직원 행동강령(etri.txt)'만을 근거로 간결하고 정확히 답한다. "
        "출처가 모호하면 '문서 근거 불충분'이라고 명시한다. 가능한 경우 장/절/조를 함께 표기한다."
    )

    print("Initializing Ray...")
    ray.init(address="auto", ignore_reinit_error=True)
    print("Ray connected successfully.")
    print("Ray version:", ray.__version__)

    print("Loading Vector Store and retrieving context...")
    vs = load_vectorstore(FAISS_DIR, EMBED_MODEL)
    ctx_q1 = retrieve_context(vs, q1, k=TOP_K)
    ctx_q2 = retrieve_context(vs, q2, k=TOP_K)
    user_q1 = f"[컨텍스트]\n{ctx_q1}\n\n[질문]\n{q1}"
    user_q2 = f"[컨텍스트]\n{ctx_q2}\n\n[질문]\n{q2}"
    print("Context retrieved.")

    run_serve_benchmark(user_q1, user_q2, sys_msg)

    ray.shutdown()
    print("\n✅ Ray has been shut down.")

if __name__ == "__main__":
    main()
