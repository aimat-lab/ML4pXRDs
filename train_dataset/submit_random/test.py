import ray

#ray.init(address='ray://10.0.0.114:10001')
ray.init(address='auto')

print(ray.cluster_resources())
