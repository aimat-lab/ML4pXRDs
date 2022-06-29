import numpy as np
from lmfit import Parameters
from lmfit import Model

xs = np.linspace(0, 5, 100)
ys = np.cos(xs) + np.random.rand(100) * 0.1


def fit_fn(x, **args):

    values = list(args.values())

    return values[0] * np.cos(x) + values[1]


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
