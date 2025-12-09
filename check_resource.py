import ray
ray.init(address="auto")
print(ray.nodes())      # 노드 리스트
print(ray.cluster_resources())  # 전체 리소스
