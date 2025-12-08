# Libraries
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 경로설정
FAISS_DIR = Path("./faiss_index")
EMBED_MODEL = "jhgan/ko-sroberta-multitask"

Q1_PATH = Path("./models/qwen25_3b")               
Q2_PATH = Path("./models/qwen25_1_5b_instruct")

TOP_K = 4
MAX_NEW_TOKENS = 1024 # 512는 끊김

# Vector Store loading (vs)
def load_vectorstore(faiss_dir: Path, embed_model: str) -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)

# Context Top k 만큼 가져오기
def retrieve_context(vs: FAISS, query: str, k: int = TOP_K) -> str:
    docs = vs.similarity_search(query, k=k)
    # 답변 간결성을 위해 본문만 연결
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


# Inference with Chat Template
@torch.inference_mode()
def chat_generate(tok, model, system_msg: str, user_msg: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    input_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,                  # greedy
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    input_len = input_ids.shape[-1]
    gen_tokens = outputs[0, input_len:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()

# Main
def main():
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

    # Load model
    tok1, model1 = load_local_llm(Q1_PATH)
    tok2, model2 = load_local_llm(Q2_PATH)

    # 동시 실행 (분산처리용)
    def run_q1():
        ans = chat_generate(tok1, model1, sys_msg, user_q1)
        return ("Qwen2.5-3B", q1, ans)

    def run_q2():
        ans = chat_generate(tok2, model2, sys_msg, user_q2)
        return ("Qwen2.5-1.5B-Instruct", q2, ans)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(run_q1), ex.submit(run_q2)]
        for fut in as_completed(futures):  # 완료된 순서대로 출력 (동시에 제출됨)
            tag, q, a = fut.result()
            print(f"[{tag}]")
            print("Q:", q)
            print("A:", a)
            print()

if __name__ == "__main__":
    main()