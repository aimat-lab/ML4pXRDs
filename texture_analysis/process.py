import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_file = "data/XRD-8 component systems.csv"
df = pd.read_csv(test_file, sep=";")

xs = np.array(df.iloc[:, list(range(0, len(df.columns.values), 2))])
ys = np.array(df.iloc[:, list(range(1, len(df.columns.values), 2))])

names = df.columns.values[1::2]

plt.plot(xs[:, 0], ys[:, 0])
plt.show()
