import matplotlib.pyplot as plt

# Use computer modern font (sans-serif)
plt.rcParams["text.usetex"] = True
# plt.rcParams["text.latex.preamble"] = [r"\usepackage[cm]{sfmath}"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("ytick", labelsize=8)
plt.rc("xtick", labelsize=8)
plt.rc("axes", labelsize=8)
plt.rc("legend", **{"fontsize": 8})

pub_width = 5.9

if __name__ == "__main__":
    plt.xlabel("test")
    plt.plot([1, 2, 3], [3, 2, 1])
    plt.show()
