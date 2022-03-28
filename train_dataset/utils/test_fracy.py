import numpy as np

if __name__ == "__main__":

    NO_wyckoffs = 10
    N_splits = 3

    a = []
    for i in range(0, N_splits - 1):
        while True:
            chosen_frac = np.random.randint(1, NO_wyckoffs)
            if chosen_frac not in a:
                a.append(chosen_frac)
                break
            else:
                continue

    a = sorted(a)

    N_per_element = np.append(a, NO_wyckoffs) - np.insert(a, 0, 0)
    assert np.sum(N_per_element) == NO_wyckoffs  # important for break condition below

    print(N_per_element)
