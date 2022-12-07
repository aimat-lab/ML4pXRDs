import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Define some sensible matplotlib default parameters that can be imported / used in other places

jobid = os.getenv("SLURM_JOB_ID")
if jobid is None or jobid == "":
    mpl.rcParams["text.usetex"] = True
    # Use computer modern font (sans-serif)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["mathtext.fontset"] = "cm"

plt.rc("ytick", labelsize=8)
plt.rc("xtick", labelsize=8)
plt.rc("axes", labelsize=8)
plt.rc("legend", **{"fontsize": 8})

pub_width = 5.9

if __name__ == "__main__":
    plt.xlabel("F")
    plt.ylabel(r"$\frac{1}{\mathrm{lol}}$")
    plt.plot([1, 2, 3], [3, 2, 1])
    plt.show()
