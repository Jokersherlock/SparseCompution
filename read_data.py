import numpy as np

# 尝试使用 genfromtxt，并指定分隔符，如果不清楚分隔符，可以尝试手动设置
data = np.genfromtxt('softmax_o.txt', delimiter=',', dtype=None, invalid_raise=False)

print(data)
