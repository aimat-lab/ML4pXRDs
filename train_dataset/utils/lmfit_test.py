import numpy as np
from lmfit import Parameters
from lmfit import Model

xs = np.linspace(0, 5, 10)
ys = np.cos(xs) + np.random.rand(10)


def fit_fn(x, **args):

    dict_keys = list(args.keys())
    print(dict_keys)

    return args[dict_keys[0]] * np.cos(x) + args[dict_keys[1]]


params = Parameters()
params.add("test3", 0.0)
params.add("test1", 0.0)
params.add("test2", 0.0)

model = Model(fit_fn)

result = model.fit(
    ys,
    x=xs,
    params=params,
)

print(result.best_values)
