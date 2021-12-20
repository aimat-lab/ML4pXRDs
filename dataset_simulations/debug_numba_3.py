import numba
import numpy as np
from numba.typed import List
import random

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

"""
@numba.njit
def test():
    test = np.array([1.0, 2.2])

    testing = np.concatenate((test, np.array([1.0])))

    print(testing)


# test()


@numba.njit
def hallo():
    return np.min(np.array([1, 2, 3], dtype=numba.types.float64))


print(hallo())
"""


# @numba.njit
# def hallo():
#    if np.random.random() > 0.5:
#        return None
#    else:
#        return [1, 2, 3]


# hallo()


@numba.njit
def apply_ops(coord, ops):

    affine_point = np.concatenate((coord, np.array([1.0])))

    results = np.empty((len(ops), 3), numba.types.float64)

    for i, op in enumerate(ops):
        results[i, :] = (op @ affine_point)[:-1]

    return results


@numba.njit
def merge(
    self_wp,
    index_i,
    index_j,
    pt,
    lattice,
    tol,
    wyckoffs_organized,
):
    """
    Given a list of fractional coordinates, merges them within a given
    tolerance, and checks if the merged coordinates satisfy a Wyckoff
    position.

    Args:
        pt: the original point (3-vector)
        lattice: a 3x3 matrix representing the unit cell
        tol: the cutoff distance for merging coordinates
        orientations: the valid orientations for a given molecule.

    Returns:
        pt: 3-vector after merge
        wp: a `pyxtal.symmetry.Wyckoff_position` object, If no matching WP, returns False.
        valid_ori: the valid orientations after merge

    """

    wp = List()
    for item in self_wp:
        wp.append(item.copy())

    # wp = deepcopy(self)
    # wp = [
    #    [[subsubitem for subsubitem in subitem] for subitem in item] for item in self_wp
    # ]  # copy symmetry operations

    pt = project(wp, pt, lattice)
    coor = apply_ops(pt, wp)

    # Main loop for merging multiple times
    while True:

        # Check distances of current WP. If too small, merge

        # test = np.expand_dims(coor[0], axis=0)
        # compare = np.array([coor[0]])

        dm = distance_matrix(np.expand_dims(coor[0], axis=0), coor, lattice)

        passed_distance_check = True
        x = np.argwhere(dm < tol)
        for y in x:
            # Ignore distance from atom to itself
            if y[0] == 0 and y[1] == 0:
                pass
            else:
                passed_distance_check = False
                break

        # for molecular crystal, one more check
        # if not check_images([coor[0]], [6], lattice, tol_matrix, PBC=PBC):
        #    passed_distance_check = False

        if not passed_distance_check:
            # mult1 = group[index].multiplicity
            mult1 = len(wyckoffs_organized[index_i][index_j])

            # Find possible wp's to merge into
            possible = []
            for i, wp0 in enumerate(wyckoffs_organized):
                mult2 = len(wp0[0])

                if (mult2 < mult1) and (mult1 % mult2 == 0):
                    for j in range(len(wp0)):
                        possible.append((i, j))
            if len(possible) == 0:
                return (
                    np.array(
                        [0.0, 0.0, 0.0], dtype=numba.types.float64
                    ),  # return something, so numba is happy
                    wyckoffs_organized[0][0],  # return something, so numba is happy
                    0,
                    0,
                    False,
                )

            # Calculate minimum separation for each WP
            distances = []
            for i, j in possible:
                wp = wyckoffs_organized[i][j]

                projected_pt = project(wp, pt.copy(), lattice)
                d = distance(pt - projected_pt, lattice)
                # distances.append(np.min(d))
                distances.append(d)

            # Choose wp with shortest translation for generating point
            tmpindex = np.argmin(np.array(distances, dtype=numba.types.float64))
            index_i, index_j = possible[tmpindex]
            wp = wyckoffs_organized[index_i][index_j]

            pt = project(wp, pt, lattice)
            coor = apply_ops(pt, wp)

        # Distances were not too small; return True
        else:
            return pt, wp, index_i, index_j, True


@numba.njit
def generate_point(ltype):

    point = np.random.rand(3)
    if ltype in ["spherical", "ellipsoidal"]:
        # Choose a point within an octant of the unit sphere
        while point.dot(point) > 1:  # squared
            point = np.random.random(3)
        # Randomly flip some coordinates
        for index in range(len(point)):
            # Scale the point by the max radius
            if random.uniform(0, 1) < 0.5:
                point[index] *= -1
    else:
        for i, a in enumerate([1, 1, 1]):
            if not a:
                if ltype in ["hexagonal", "trigonal", "rhombohedral"]:
                    point[i] *= 1.0 / np.sqrt(3.0)
                else:
                    point[i] -= 0.5
    return point


@numba.njit
def choose_wyckoff(wyckoffs_organized, number):
    """
    Choose a Wyckoff position to fill based on the current number of atoms
    needed to be placed within a unit cell
    Rules:
        0) use the pre-assigned list if this is provided
        1) The new position's multiplicity is equal/less than (number).
        2) We prefer positions with large multiplicity.

    Args:
        group: a pyxtal.symmetry.Group object
        number: the number of atoms still needed in the unit cell
        site: the pre-assigned Wyckoff sites (e.g., 4a)

    Returns:
        Wyckoff position. If no position is found, returns False
    """

    # if site is not None:
    #    raise Exception("Self-specifying wyckoff sites is not currently supported.")
    #    return site  # only specify the Wyckoff position by the index; only indices supported as preoccupied inputs
    # else:

    if random.uniform(0, 1) > 0.5:  # choose from high to low
        for i, wyckoff in enumerate(wyckoffs_organized):
            if len(wyckoff[0]) <= number:
                chosen = random.randint(0, len(wyckoff) - 1)
                return True, wyckoff[chosen], i, chosen

        return (
            False,
            wyckoffs_organized[0][0],
            0,
            0,
        )  # return something when not valid, so numba is happy
    else:
        good_wyckoff = []
        good_indices = []
        for i, wyckoff in enumerate(wyckoffs_organized):
            if len(wyckoff[0]) <= number:
                for j, w in enumerate(wyckoff):
                    good_wyckoff.append(w)
                    good_indices.append((i, j))

        if len(good_wyckoff) > 0:
            chosen = random.randint(0, len(good_wyckoff) - 1)
            return (
                True,
                good_wyckoff[chosen],
                good_indices[chosen][0],
                good_indices[chosen][1],
            )
        else:
            return (
                False,
                wyckoffs_organized[0][0],
                0,
                0,
            )  # return something when not valid, so numba is happy


@numba.njit
def test(
    wyckoffs_organized,
):

    # numba workaround, so that it knows the type of this list:
    wyckoff_sites_tmp = List(
        [
            (
                wyckoffs_organized[0][0],
                0,
                0,
                np.array([1.1, 2.2, 3.3], dtype=numba.types.float64),
                "Os",
                np.array([[1.1, 2.2, 3.3], [3.3, 2.2, 1.1]], dtype=numba.types.float64),
            )
        ]
    )

    valid, wp, i, j = choose_wyckoff(wyckoffs_organized, 4)

    pt = generate_point(ltype)

    # Merge coordinates if the atoms are close
    pt, wp, i, j, worked = merge(wp, i, j, pt, cell, tol, wyckoffs_organized)

    # Use a Wyckoff_site object for the current site
    # new_site = atom_site(wp, pt, specie)
    new_site = (wp, i, j, pt, "O", apply_ops(pt, wp))

    wyckoff_sites_tmp.append(new_site)


wyckoffs_organized = List()
for i in range(0, 3):
    newList = List()
    for j in range(0, 3):
        newnewList = List()
        for k in range(0, 3):
            newnewList.append(np.random.random(size=(4, 4)))
        newList.append(newnewList)
    wyckoffs_organized.append(newList)

test(wyckoffs_organized)
