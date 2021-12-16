import numba
import numpy as np

deg = 180.0 / np.pi
rad = np.pi / 180.0


@numba.njit
def angle(v1, v2, radians=True):
    """
    Calculate the angle (in radians) between two vectors.

    Args:
        v1: a 1x3 vector
        v2: a 1x3 vector
        radians: whether to return angle in radians (default) or degrees

    Returns:
        the angle in radians between the two vectors
    """
    v1 = np.real(v1)
    v2 = np.real(v2)
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # if np.isclose(dot, 1.0):
    if np.abs(dot - 1) < 1e-3:
        a = 0
    elif np.abs(dot + 1) < 1e-3:
        a = np.pi
    else:
        a = np.arccos(dot)

    if radians:
        return a
    else:
        return a * deg


@numba.njit
def matrix2para(matrix, radians=True):
    """
    Given a 3x3 matrix representing a unit cell, outputs a list of lattice
    parameters.

    Args:
        matrix: a 3x3 array or list, where the first, second, and third rows
            represent the a, b, and c vectors respectively
        radians: if True, outputs angles in radians. If False, outputs in
            degrees

    Returns:
        a 1x6 list of lattice parameters [a, b, c, alpha, beta, gamma]. a, b,
        and c are the length of the lattice vectos, and alpha, beta, and gamma
        are the angles between these vectors (in radians by default)
    """
    cell_para = np.zeros(6)
    # a
    cell_para[0] = np.linalg.norm(matrix[0])
    # b
    cell_para[1] = np.linalg.norm(matrix[1])
    # c
    cell_para[2] = np.linalg.norm(matrix[2])
    # alpha
    cell_para[3] = angle(matrix[1], matrix[2])
    # beta
    cell_para[4] = angle(matrix[0], matrix[2])
    # gamma
    cell_para[5] = angle(matrix[0], matrix[1])

    if not radians:
        # convert radians to degrees
        deg = 180.0 / np.pi
        cell_para[3] *= deg
        cell_para[4] *= deg
        cell_para[5] *= deg
    return cell_para


@numba.njit
def gaussian(min, max, sigma=3.0):
    """
    Choose a random number from a Gaussian probability distribution centered
    between min and max. sigma is the number of standard deviations that min
    and max are away from the center. Thus, sigma is also the largest possible
    number of standard deviations corresponding to the returned value. sigma=2
    corresponds to a 95.45% probability of choosing a number between min and
    max.

    Args:
        min: the minimum acceptable value
        max: the maximum acceptable value
        sigma: the number of standard deviations between the center and min or max

    Returns:
        a value chosen randomly between min and max
    """

    center = (max + min) * 0.5
    delta = np.fabs(max - min) * 0.5
    ratio = delta / sigma
    while True:
        x = np.random.normal(scale=ratio, loc=center)
        if x > min and x < max:
            return x


@numba.njit
def random_vector(width=0.35, unit=False):
    """
    Generate a random vector for lattice constant generation. The ratios between
    x, y, and z of the returned vector correspond to the ratios between a, b,
    and c. Results in a Gaussian distribution of the natural log of the ratios.

    Args:
        minvec: the bottom-left-back minimum point which can be chosen
        maxvec: the top-right-front maximum point which can be chosen
        width: the width of the normal distribution to use when choosing values.
            Passed to np.random.normal
        unit: whether or not to normalize the vector to determinant 1

    Returns:
        a 1x3 numpy array of floats
    """

    vec = np.array(
        [
            np.exp(np.random.normal(0.0, width)),
            np.exp(np.random.normal(0.0, width)),
            np.exp(np.random.normal(0.0, width)),
        ],
        # dtype=numba.types.float64,
    )
    if unit:
        return vec / np.linalg.norm(vec)
    else:
        return vec


@numba.njit
def random_shear_matrix(width=1.0, unitary=False):
    """
    Generate a random symmetric shear matrix with Gaussian elements. If unitary
    is True, normalize to determinant 1

    Args:
        width: the width of the normal distribution to use when choosing values.
            Passed to np.random.normal
        unitary: whether or not to normalize the matrix to determinant 1

    Returns:
        a 3x3 numpy array of floats
    """

    mat = np.zeros((3, 3))
    determinant = 0
    while determinant == 0:
        a, b, c = (
            np.random.normal(0.0, width),
            np.random.normal(0.0, width),
            np.random.normal(0.0, width),
        )
        mat = np.array([[1, a, b], [a, 1, c], [b, c, 1]])
        determinant = np.linalg.det(mat)
    if unitary:
        # new = mat / np.cbrt(np.linalg.det(mat))
        new = mat / (np.linalg.det(mat)) ** (1 / 3)
        return new
    else:
        return mat


@numba.njit
def generate_lattice(
    ltype,
    volume,
    minvec=1.2,
    minangle=np.pi / 6,
    max_ratio=10.0,
    maxattempts=100,
):

    print(repr(locals()))

    maxangle = np.pi - minangle
    for n in range(maxattempts):
        # Triclinic
        # if sg <= 2:
        if ltype == "triclinic":
            # Derive lattice constants from a random matrix
            mat = random_shear_matrix(width=0.2)
            a, b, c, alpha, beta, gamma = matrix2para(mat)
            x = np.sqrt(
                1
                - np.cos(alpha) ** 2
                - np.cos(beta) ** 2
                - np.cos(gamma) ** 2
                + 2 * (np.cos(alpha) * np.cos(beta) * np.cos(gamma))
            )
            vec = random_vector()
            abc = volume / x
            xyz = vec[0] * vec[1] * vec[2]
            a = vec[0] * np.cbrt(abc) / np.cbrt(xyz)
            b = vec[1] * np.cbrt(abc) / np.cbrt(xyz)
            c = vec[2] * np.cbrt(abc) / np.cbrt(xyz)
        # Monoclinic
        elif ltype in ["monoclinic", "Monoclinic"]:
            alpha, gamma = np.pi / 2, np.pi / 2
            beta = gaussian(minangle, maxangle)
            x = np.sin(beta)
            vec = random_vector()
            xyz = vec[0] * vec[1] * vec[2]
            abc = volume / x
            a = vec[0] * np.cbrt(abc) / np.cbrt(xyz)
            b = vec[1] * np.cbrt(abc) / np.cbrt(xyz)
            c = vec[2] * np.cbrt(abc) / np.cbrt(xyz)
        # Orthorhombic
        # elif sg <= 74:
        elif ltype in ["orthorhombic", "Orthorhombic"]:
            alpha, beta, gamma = np.pi / 2, np.pi / 2, np.pi / 2
            x = 1
            vec = random_vector()
            xyz = vec[0] * vec[1] * vec[2]
            abc = volume / x
            a = vec[0] * np.cbrt(abc) / np.cbrt(xyz)
            b = vec[1] * np.cbrt(abc) / np.cbrt(xyz)
            c = vec[2] * np.cbrt(abc) / np.cbrt(xyz)
        # Tetragonal
        # elif sg <= 142:
        elif ltype in ["tetragonal", "Tetragonal"]:
            alpha, beta, gamma = np.pi / 2, np.pi / 2, np.pi / 2
            x = 1
            vec = random_vector()
            c = vec[2] / (vec[0] * vec[1]) * np.cbrt(volume / x)
            a = b = np.sqrt((volume / x) / c)
        # Trigonal/Rhombohedral/Hexagonal
        # elif sg <= 194:
        elif ltype in ["hexagonal", "trigonal", "rhombohedral"]:
            alpha, beta, gamma = np.pi / 2, np.pi / 2, np.pi / 3 * 2
            x = np.sqrt(3.0) / 2.0
            vec = random_vector()
            c = vec[2] / (vec[0] * vec[1]) * np.cbrt(volume / x)
            a = b = np.sqrt((volume / x) / c)
        # Cubic
        # else:
        elif ltype in ["cubic", "Cubic"]:
            alpha, beta, gamma = np.pi / 2, np.pi / 2, np.pi / 2
            s = (volume) ** (1.0 / 3.0)
            a, b, c = s, s, s

        # Check that lattice meets requirements
        maxvec = (a * b * c) / (minvec ** 2)

        # Define limits on cell dimensions
        """
        if "min_l" not in kwargs:
            min_l = minvec
        else:
            min_l = kwargs["min_l"]
        if "mid_l" not in kwargs:
            mid_l = min_l
        else:
            mid_l = kwargs["mid_l"]
        if "max_l" not in kwargs:
            max_l = mid_l
        else:
            max_l = kwargs["max_l"]
        """
        # Defining cell dimensions not currently supported!
        min_l = minvec
        mid_l = minvec
        max_l = minvec

        l_min = min(a, b, c)
        l_max = max(a, b, c)
        for x in (a, b, c):
            if x <= l_max and x >= l_min:
                l_mid = x
        if not (l_min >= min_l and l_mid >= mid_l and l_max >= max_l):
            continue

        if minvec < maxvec:
            # Check minimum Euclidean distances
            smallvec = min(
                a * np.cos(max(beta, gamma)),
                b * np.cos(max(alpha, gamma)),
                c * np.cos(max(alpha, beta)),
            )
            if (
                a > minvec
                and b > minvec
                and c > minvec
                and a < maxvec
                and b < maxvec
                and c < maxvec
                and smallvec < minvec
                and alpha > minangle
                and beta > minangle
                and gamma > minangle
                and alpha < maxangle
                and beta < maxangle
                and gamma < maxangle
                and a / b < max_ratio
                and a / c < max_ratio
                and b / c < max_ratio
                and b / a < max_ratio
                and c / a < max_ratio
                and c / b < max_ratio
            ):
                return np.array([a, b, c, alpha, beta, gamma])

    # If maxattempts tries have been made without success
    msg = "Could not get lattice after {:d} cycles for volume {:.2f}".format(
        maxattempts, volume
    )
    raise ValueError(msg)
    # return


generate_lattice("monoclinic", 737.1810951578468)
