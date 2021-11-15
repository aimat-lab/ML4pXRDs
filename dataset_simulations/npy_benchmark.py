import numpy as np
import time

current = 3

if current == 0:

    all = []
    for i in range(0, 5000):
        all.append(None)

    for i in range(0, 5000):
        all.append([])

    all = np.array(all, dtype=object)

    start = time.time()
    np.save("test.npy", all)
    print(f"Saving: {time.time() - start}")

elif current == 1:

    test = np.load("test.npy", allow_pickle=True)

    for i in range(0, 10000):
        new = np.random.uniform(0, 1, 9001)
        test[i] = new

    test = np.array(test.tolist(), float)

    np.save("test.npy", test)

elif current == 2:

    start = time.time()
    test = np.load("test.npy")
    print(f"Loading: {time.time() - start}")

    pass

elif current == 3:

    start = time.time()
    test = np.load("test.npy", mmap_mode="r", allow_pickle=True)
    print(test[1:1000])
    print(f"Loading mmap: {time.time() - start}")

elif current == 4:

    test = np.empty(
        (
            100000,
            9001,
        )
    )
    test[:, :] = np.nan

    time.sleep(5)
