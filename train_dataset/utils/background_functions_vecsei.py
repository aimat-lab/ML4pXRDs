import numpy as np
import matplotlib.pyplot as plt

start_x = 5
end_x = 90
N = 8501
pattern_x = np.linspace(start_x, end_x, N)

########## Peak profile functions from Vecsei ##########


def theta_rel(theta):
    return (theta - start_x) / (end_x - start_x)


def f_step_up(theta, h, alpha, theta_rel_step):
    return h * (1 / (1 + np.exp(alpha * (theta_rel(theta) - theta_rel_step))))


def f_step_down(theta, h, alpha, theta_rel_step):
    return h * (1 - (1 / (1 + np.exp(alpha * (theta_rel(theta) - theta_rel_step)))))


def f_polynomial(theta, n_max, alpha_n):
    return np.abs(
        np.sum([alpha_n[i] * theta_rel(theta) ** i for i in range(0, n_max + 1)])
    )


def f_bump(theta, h, n):
    return h * (n * theta_rel(theta)) ** 2 * np.exp(-1 * n * theta_rel(theta))


def trunc_normal(min, max, mean, std):
    while True:
        value = np.random.normal(mean, std)

        if value > min and value < max:
            return value
        else:
            continue


def generate_background_vecsei():

    diffractogram = np.zeros(N)

    # TODO: find potentially better way of handling noise
    # patterns go from 0 to 1
    # diffractogram += np.random.uniform(0.0002, 0.002, N)

    # TODO: To introduce noise, we augment our XRD patterns with a random signal drawn
    # from a uniform distribution U (0.002, 0.02) between 0.002 and 0.02.

    choices = [1 if x < 0.5 else 0 for x in np.random.uniform(size=4)]
    T = 0.1 / np.sum(choices)

    if choices[0]:
        diffractogram += f_step_up(
            pattern_x,
            trunc_normal(0, T, T / 3, T / 7),
            np.random.uniform(10, 60),
            np.random.uniform(0, 1 / 7),
        )

    if choices[1]:
        diffractogram += f_step_down(
            pattern_x,
            trunc_normal(0, T, T / 3, T / 7),
            np.random.uniform(10, 60),
            np.random.uniform(1 - 1 / 7, 1),
        )

    if choices[2]:
        n_max = np.random.randint(0, 5)
        alpha_n = np.zeros(n_max + 1)
        for i, alpha in enumerate(alpha_n):
            if np.random.uniform() < 0.5:
                alpha_n[i] = 3 * T / (2 * (n_max + 1)) * np.random.uniform(-1, 1)

        diffractogram += f_polynomial(pattern_x, n_max, alpha_n)

    if choices[3]:
        diffractogram += f_bump(
            pattern_x,
            trunc_normal(0, 3 * T / 5, 2 * T / 5, 3 * T / 35),
            np.random.uniform(40, 70),
        )

    return diffractogram

    # TODO: Add the theoretical pattern (let it be passed by argument)
    # using the signal to noise ratio given in the paper


##########

if __name__ == "__main__":

    for i in range(0, 100):
        diff = generate_background_vecsei()

        plt.plot(pattern_x, diff)
        plt.show()
