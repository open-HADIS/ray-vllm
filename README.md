Ray–vLLM 기반 분산 추론 프레임워크
Distributed Inference Framework with Ray & vLLM for RAG-based LLM Serving

본 프로젝트는 Ray 기반 클러스터 자원 스케줄링과
vLLM의 Paged Attention 기반 KV 캐시 최적화를 결합하여
대규모 LLM + RAG 서비스의 지연시간(TTFT, Latency) 감소 및 처리량(TPS) 극대화를 목적으로 한다.

동일한 하드웨어 환경에서 다음 네 가지 구조를 정량 비교 평가한다.

Single Node (Baseline)

Ray Actor Cluster (RAC)

Ray Serve Cluster (RSC)

Ray + vLLM Cluster (RVC)

본 프로젝트는 다음 논문의 실험 구조를 기반으로 한다.

Performance Analysis of a Ray–vLLM Based Distributed System for Efficient LLM Inference
UST–ETRI, 2025

📑 Table of Contents

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

1. 전체 시스템 아키텍처
User Query
   ↓
RAG Pipeline (FAISS + ko-sroberta)
   ↓
Prompt Expansion
   ↓
Distributed Inference
   ↓
LLM Output

2. 구성 요소

RAG: FAISS + ko-sroberta-multitask

LLM: Qwen2.5-3B, Qwen2.5-1.5B-Instruct

Distributed Runtime: Ray (Multi-node, Multi-GPU)

Inference Optimizer: vLLM (Paged Attention, Continuous Batching)

3. 성능 평가 지표 (KPM)
지표	설명
TTFT	Time To First Token
Latency_avg	요청 1개당 평균 처리 지연
Latency_total	라운드 전체 벽시계 지연
TPS	초당 처리 토큰 수

모든 실험은 동일한 질문, 동일한 RPS, 동일한 모델 조건에서 수행된다.

4. 실험 코드 구성
파일명	설명
baseline.py	단일 노드 멀티스레드 기반 추론
baseline_fetched.py	Baseline 개선판 (n=1에서 Q1+Q2 동시 처리)
baseline_quality_check.py	Baseline 추론 결과 품질 검증
RayCluster_fetched.py	Ray Actor 기반 분산 추론
RayServe_fetched.py	Ray Serve 기반 서빙 구조
RayVllm_fetched.py	Ray + vLLM 통합 분산 추론 구조 (최종 모델)
5. 실행 환경
5.1 필수 라이브러리 설치
pip install torch transformers langchain faiss-cpu ray vllm

5.2 공통 환경 변수 (오프라인 모드)
export HF_HOME=/mnt/shared/hf-home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

5.3 디렉토리 구조
/mnt/shared
 ├── faiss_index
 └── models
     ├── ko-sroberta-multitask
     ├── qwen25_3b
     └── qwen25_1_5b_instruct

6. 실행 방법
6.1 Baseline (Single Node)
python baseline.py


또는 개선판:

python baseline_fetched.py

6.2 Ray Actor Cluster (RAC)
ray start --head
ray start --address=<HEAD_NODE_IP>:6379
python RayCluster_fetched.py

6.3 Ray Serve Cluster (RSC)
ray start --head
python RayServe_fetched.py

6.4 Ray + vLLM Cluster (RVC, 최종 구조)
ray start --head
python RayVllm_fetched.py

7. vLLM 튜닝 옵션
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEM_UTIL=0.69
export VLLM_KV_CACHE_DTYPE=auto
export VLLM_QUANT=awq

8. 분산 구조별 핵심 차이
구조	특징
Baseline	단일 노드, Python Thread 기반
Ray Actor	GPU 1개당 Actor 1개, 명시적 스케줄링
Ray Serve	HTTP 기반 서빙, Replica 분산
Ray + vLLM	PagedAttention + Continuous Batching + KV Cache 최적화
9. 핵심 실험 결과 요약 (논문 기준)

Ray + vLLM 구조(RVC)는 Ray Actor 대비 다음 성능 향상을 달성하였다.

지표	개선률
TTFT	88.49% 감소
평균 Latency	72.99% 감소
전체 Latency	70.97% 감소
TPS	171.18% 증가

고부하(RPS ≥ 24) 환경에서 연속 배칭 + KV 캐시 페이징 효과 극대화

메모리 단편화 감소 및 GPU 유휴 시간 최소화

10. 프로젝트 목적 요약

Ray는 클러스터 전체 자원 스케줄링을 담당하고,
vLLM은 노드 내부 추론을 극단적으로 최적화한다.

이 둘의 결합이 현재 구조 중
가장 강력한 LLM 분산 서빙 해법임을 실증한다.

11. 향후 확장

Docker 기반 Ray + vLLM 배포 스택

gRPC 기반 실시간 서빙 API

Multi-LLM 파이프라인 자동 분산 스케줄링

NPU 기반 vLLM 런타임 이식

12. Citation
@article{kang2025rayvllm,
  title={Performance Analysis of a Ray–vLLM Based Distributed System for Efficient LLM Inference},
  author={Kang, Minsu and Kim, Young-Joo and Kim, Seon-Tae},
  journal={UST-ETRI},
  year={2025}
}
