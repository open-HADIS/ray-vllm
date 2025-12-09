from pathlib import Path
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

TXT_PATH = Path("/home/etriray-head/apps/rag_ray/etri.txt")   # 원본 텍스트 파일
FAISS_DIR = Path("./faiss_index")                             # 저장할 DB 디렉토리
EMBED_MODEL = "jhgan/ko-sroberta-multitask"                   # 한국어 특화 임베딩 모델 (변경 가능)

def build_faiss_index(txt_path: Path, faiss_dir: Path, embed_model: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,      # 청크길이
        chunk_overlap=50,    # 중복부분
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_text(raw_text)
    docs = [Document(page_content=t) for t in texts]

    # Model
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    # FAISS DB
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_dir)
    print(f"{faiss_dir} 에 FAISS index 저장 완료")

if __name__ == "__main__":
    build_faiss_index(TXT_PATH, FAISS_DIR, EMBED_MODEL)