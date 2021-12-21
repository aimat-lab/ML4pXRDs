import numpy as np
import random
import numba


@numba.njit
def test():
    np.random.seed(123)
    random.seed(123)

    print(np.random.randint(0, 100))
    print(np.random.randint(0, 100))
    print(np.random.randint(0, 100))

    print(np.random.random())
    print(np.random.random())
    print(np.random.random())

    print(np.random.random())
    print(np.random.random())
    print(np.random.random())

    print(np.random.uniform(0, 1))

    print(np.random.normal(scale=0.5, loc=0.0))
    print(np.random.normal(0.5, 0.5))

    print(np.random.rand())

    print(np.random.random(3))


test()
