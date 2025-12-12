# -*- coding: utf-8 -*-
# === Fair Comparison: Original (Transformers) vs. Quantized (vLLM) RAG ===

import os
from pathlib import Path
import ray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from vllm import LLM, SamplingParams
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import threading

# -------------------- Configuration --------------------
os.environ.setdefault("HF_HOME", "/mnt/shared/hf-home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# --- 비교할 모델 경로 ---
ORIGINAL_MODEL_PATH = Path("/mnt/shared/models/qwen25_3b")
# W4A16 또는 FP8 압축 모델 경로를 지정하세요.
COMPRESSED_MODEL_PATH = Path("/mnt/shared/models/qwen25_3b-W4A16-compressed")

# --- RAG 설정 ---
FAISS_DIR = Path("/mnt/shared/faiss_index")
EMBED_MODEL = "/mnt/shared/models/ko-sroberta-multitask"
TOP_K = 4
MAX_NEW_TOKENS = 1024

# -------------------- Utility Functions --------------------
def load_vectorstore(faiss_dir: Path, embed_model: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=embed_model, local_files_only=True)
    return FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)

def retrieve_context(vs: FAISS, query: str, k: int = TOP_K) -> str:
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content.strip() for d in docs)

# ==============================================================================
# Actor Definitions for Fair Comparison
# ==============================================================================

# Actor 1: 원본 모델을 위한 Transformers 기반 워커
@ray.remote(num_gpus=1)
class TransformersModelWorker:
    def __init__(self, model_dir: str):
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", torch_dtype="auto", local_files_only=True
        )

    @torch.inference_mode()
    def generate_text(self, system_msg: str, user_msg: str) -> str:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        input_ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id,
            streamer=streamer, repetition_penalty=1.1
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        chunks = [piece for piece in streamer]
        thread.join()
        return "".join(chunks).strip()

# Actor 2: 압축 모델을 위한 vLLM 기반 워커
@ray.remote(num_gpus=1)
class VLLMModelWorker:
    def __init__(self, model_dir: str):
        self.llm = LLM(model=model_dir, trust_remote_code=True, gpu_memory_utilization=0.4)
        self.tokenizer = self.llm.get_tokenizer()

    def generate_text(self, system_msg: str, user_msg: str) -> str:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # `temperature=0`는 `do_sample=False`와 유사한 결정론적 출력을 유도합니다.
        sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=1.1)
        outputs = self.llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text.strip()

# ==============================================================================
# Main Comparison Logic
# ==============================================================================
def main():
    print("Connecting to Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)
    print("Connected.")

    # 비교에 사용할 동일한 RAG 컨텍스트 준비
    q1 = "상급자의 부당한 지시를 받은 경우 하급자가 취할 수 있는 절차를 단계별로 나열해줘."
    sys_msg = "너는 'ETRI 임직원 행동강령(etri.txt)'만을 근거로 간결하고 정확히 답한다."
    vs = load_vectorstore(FAISS_DIR, EMBED_MODEL)
    ctx_q1 = retrieve_context(vs, q1, k=TOP_K)
    user_q1 = f"[컨텍스트]\n{ctx_q1}\n\n[질문]\n{q1}"

    print("Creating actors for Original (Transformers) and Compressed (vLLM) models...")
    original_worker = TransformersModelWorker.remote(str(ORIGINAL_MODEL_PATH))
    compressed_worker = VLLMModelWorker.remote(str(COMPRESSED_MODEL_PATH))
    print("Actors are ready.")

    print("Submitting the same RAG request to both actors...")
    original_future = original_worker.generate_text.remote(sys_msg, user_q1)
    compressed_future = compressed_worker.generate_text.remote(sys_msg, user_q1)

    original_output = ray.get(original_future)
    compressed_output = ray.get(compressed_future)
    print("All requests processed.")

    # 최종 결과 비교 출력
    print("\n\n" + "="*80)
    print("              📝 RAG Generation Comparison 📝")
    print("="*80)
    print(f"\n--- Output from ORIGINAL Model (via Transformers) ---")
    print(original_output)
    print("\n" + "-"*80)
    print(f"\n--- Output from COMPRESSED Model (via vLLM) ---")
    print(compressed_output)
    print("\n" + "="*80)

    ray.shutdown()
    print("\n✅ Fair comparison finished.")

if __name__ == "__main__":
    main()