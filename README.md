
🚀 Ray–vLLM 기반 분산 추론 프레임워크 

🚀 Distributed Inference Framework with Ray & vLLM for RAG-based LLM Serving


📌 프로젝트 개요
------------------------------------------------------------
본 프로젝트는 Ray 기반 클러스터 자원 스케줄링과
vLLM의 Paged Attention 기반 KV 캐시 최적화를 결합하여,

✅ 대규모 LLM + RAG 서비스의
✅ 지연시간(TTFT, Latency) 감소
✅ 처리량(TPS) 극대화

를 목적으로 한다.


📌 비교 평가 대상 분산 구조 (4가지)
------------------------------------------------------------
① Single Node (Baseline)
② Ray Actor Cluster (RAC)
③ Ray Serve Cluster (RSC)
④ Ray + vLLM Cluster (RVC)

본 프로젝트는 다음 논문의 실험 구조를 기반으로 한다.

📄 Performance Analysis of a Ray–vLLM Based Distributed System for Efficient LLM Inference  
🏛 UST–ETRI, 2025



📑 TABLE OF CONTENTS
============================================================
1. 전체 시스템 아키텍처
2. 구성 요소
3. 성능 평가 지표 (KPM)
4. 실험 코드 구성
5. 실행 환경
6. 실행 방법
7. vLLM 튜닝 옵션
8. 분산 구조별 핵심 차이
9. 핵심 실험 결과 요약
10. 프로젝트 목적 요약
11. 향후 확장
12. Citation



✅ 1. 전체 시스템 아키텍처
============================================================

<br>
[ User Query ] <br>
↓ <br>
[ RAG Pipeline (FAISS + ko-sroberta) ] <br>
↓ <br>
[ Prompt Expansion ] <br>
↓ <br>
[ Distributed Inference ] <br>
↓ <br>
[ LLM Output ] <br>
<br>
<br>



✅ 2. 구성 요소
============================================================

🔹 RAG               : FAISS + ko-sroberta-multitask  
🔹 LLM               : Qwen2.5-3B, Qwen2.5-1.5B-Instruct  
🔹 Distributed Runtime: Ray (Multi-node, Multi-GPU)  
🔹 Inference Optimizer: vLLM (Paged Attention, Continuous Batching)



✅ 3. 성능 평가 지표 (KPM)
============================================================

<br>
지표 | 설명 <br>
TTFT | Time To First Token <br>
Latency_avg | 요청 1개당 평균 처리 지연 <br>
Latency_total | 라운드 전체 벽시계 지연 <br>
TPS | 초당 처리 토큰 수 <br>
<br>
※ 모든 실험은 <br>
✔ 동일한 질문 <br>
✔ 동일한 RPS <br>
✔ 동일한 모델 <br>
조건에서 수행된다. <br>
<br>
<br>



✅ 4. 실험 코드 구성
============================================================

<br>
baseline.py : 단일 노드 멀티스레드 기반 추론 <br>
baseline_fetched.py : Baseline 개선판 <br>
baseline_quality_check.py : Baseline 추론 결과 품질 검증 <br>
RayCluster_fetched.py : Ray Actor 기반 분산 추론 <br>
RayServe_fetched.py : Ray Serve 기반 서빙 구조 <br>
RayVllm_fetched.py : Ray + vLLM 통합 분산 추론 <br>
<br>
<br>



✅ 5. 실행 환경
============================================================

<br>
[5.1] 필수 라이브러리 설치 <br>
pip install torch transformers langchain faiss-cpu ray vllm <br>
<br>
[5.2] 공통 환경 변수 (오프라인 모드) <br>
export HF_HOME=/mnt/shared/hf-home <br>
export HF_HUB_OFFLINE=1 <br>
export TRANSFORMERS_OFFLINE=1 <br>
export HF_DATASETS_OFFLINE=1 <br>
<br>
[5.3] 디렉토리 구조 <br>
/mnt/shared <br>

faiss_index <br>

models <br>

ko-sroberta-multitask <br>

qwen25_3b <br>

qwen25_1_5b_instruct <br>

<br>
<br>


✅ 6. 실행 방법
============================================================

<br>
[6.1] Baseline (Single Node) <br>
python baseline.py <br>
또는 <br>
python baseline_fetched.py <br>
<br>
[6.2] Ray Actor Cluster (RAC) <br>
ray start --head <br>
ray start --address=<HEAD_NODE_IP>:6379 <br>
python RayCluster_fetched.py <br>
<br>
[6.3] Ray Serve Cluster (RSC) <br>
ray start --head <br>
python RayServe_fetched.py <br>
<br>
[6.4] Ray + vLLM Cluster (RVC) <br>
ray start --head <br>
python RayVllm_fetched.py <br>
<br>
<br>


✅ 7. vLLM 튜닝 옵션
============================================================

export VLLM_MAX_MODEL_LEN=4096  
export VLLM_GPU_MEM_UTIL=0.69  
export VLLM_KV_CACHE_DTYPE=auto  
export VLLM_QUANT=awq  



✅ 8. 분산 구조별 핵심 차이
============================================================

Baseline : 단일 노드, Python Thread 기반 <br>
Ray Actor : GPU 1개당 Actor 1개, 명시적 스케줄링 <br>
Ray Serve : HTTP 기반 서빙, Replica 분산 <br>
Ray + vLLM : PagedAttention + 연속 배칭/KV캐시 최적화 <br>
<br>
<br>

✅ 9. 핵심 실험 결



✅ 9. 핵심 실험 결과 요약 (논문 기준)
============================================================

TTFT : 🔻 88.49% 감소 <br>
평균 Latency : 🔻 72.99% 감소 <br>
전체 Latency : 🔻 70.97% 감소 <br>
TPS : 🔺 171.18% 증가 <br>
<br>
✔ 고부하(RPS ≥ 24) 환경에서 <br>
✔ 연속 배칭 + KV 캐시 페이징 효과 극대화 <br>
✔ 메모리 단편화 감소 및 GPU 유휴 시간 최소화 <br>
<br>
<br>


✅ 10. 프로젝트 목적 요약
============================================================

Ray는 🔧 클러스터 전체 자원 스케줄링 담당  
vLLM은 ⚡ 노드 내부 추론 극한 최적화 담당  

➡ 이 둘의 결합이
➡ 현재 구조 중 가장 강력한
➡ LLM 분산 서빙 해법임을 실증한다.



✅ 11. 향후 확장
============================================================

- 📦 Docker 기반 Ray + vLLM 배포 스택
- 🔌 gRPC 기반 실시간 서빙 API
- 🔀 Multi-LLM 파이프라인 자동 분산 스케줄링
- 🧠 NPU 기반 vLLM 런타임 이식



✅ 12. Citation
============================================================

@article{kang2025rayvllm,
  title  = {Performance Analysis of a Ray–vLLM Based Distributed System for Efficient LLM Inference},
  author = {Kang, Minsu and Kim, Young-Joo and Kim, Seon-Tae},
  journal= {UST-ETRI},
  year   = {2025}
}


