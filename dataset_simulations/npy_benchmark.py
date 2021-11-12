import numpy as np
import time

"""
all = []
for i in range(0, 100000):
    new = np.random.uniform(0, 1, 9018)
    all.append(new)

all = np.array(all)

np.save("test.npy", all)
"""

test = np.load("test.npy", mmap_mode="r")

print(test)


time.sleep(10)
