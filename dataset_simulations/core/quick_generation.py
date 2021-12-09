# The core of the following code is taken from the pyxtal package.
# It is altered and optimized to run using numba.
# If performance is not a concern to you, you should use the original pyxtal code.


from pyxtal import pyxtal
import time

# TODO:
# Pass the group object from outside into pytal (reuse)
# Pass revelant stuff (covalent radii, etc. inside)
# Create files for __init__.py and crystal.py myself
# Modify the two main functions in crystal.py

if __name__ == "__main__":

    start = time.time()

    my_crystal = pyxtal()

    my_crystal.from_random(
        dim=3,
        group=114,
        species=["Os", "Br", "Hg", "La", "P"],
        numIons=[4, 4, 2, 8, 8],
    )

    stop = time.time()
    print(f"{stop-start}")
