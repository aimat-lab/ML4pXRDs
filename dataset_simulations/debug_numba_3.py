import numba
import numpy as np
from numba.typed import List

"""
@numba.njit(
    numba.types.Tuple((numba.types.int64, numba.types.boolean,))(
        numba.types.float64[:, :],
        # numba.types.List(
        numba.types.Tuple(
            (
                numba.types.int64,
                numba.types.float64,
            )
        )[:]
        # ),
    )
)
def test(a, b):
    print(a[0, 0])

    return 5, False
"""

"""
@numba.njit(
    # numba.types.none(
    #    numba.types.ListType[numba.types.Tuple((numba.int64, numba.int64))]
    # )
    numba.types.int64(numba.types.List(numba.int64))
)
def test(a):
    print(a[0])

    return 5


@numba.njit
def testing():
    test([4])


testing()
"""

"""
@numba.njit(numba.types.void(numba.types.float64[:, :]))
def test(a):
    print(a[0, 0])


test(np.array([[0.0]]))
"""


@numba.njit
def test():
    test = np.array([1.0, 2.2])

    testing = np.concatenate((test, np.array([1.0])))

    print(testing)


# test()


def hallo():
    print()
    welt()


def welt():
    print()


hallo()
