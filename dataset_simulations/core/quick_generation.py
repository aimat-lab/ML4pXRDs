# The core of the following code is taken from the pyxtal package.
# It is altered and optimized to run using numba.
# If performance is not a concern to you, you should use the original pyxtal code.

from pyxtal import pyxtal
import time
import numpy as np
import random

# TODO:
# Pass the group object from outside into pyxtal (reuse)

if __name__ == "__main__":

    np.random.seed(3)
    random.seed(3)

    start = time.time()

    my_crystal = pyxtal()

    my_crystal.from_random(
        dim=3,
        group=114,
        species=["Os", "Br", "Hg", "La", "P"],
        numIons=[4, 4, 2, 8, 8],
    )

    for site in my_crystal.atom_sites:
        # print(site.coords)
        print(site.coords)

        pyxtal

    stop = time.time()
    print(f"{stop-start}")
