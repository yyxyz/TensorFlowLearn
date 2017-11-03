import numpy as np

A = np.arange(2,14).reshape((3,4))
print(A)
print()

print(np.argmin(A))
print(np.argmax(A))
print("计算平均值")
print(np.mean(A))
print(A.mean())
print(np.average(A))
print("计算中位数")
print(np.median(A))
print("计算累加值")
print(np.cumsum(A))





