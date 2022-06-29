import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

xs = np.linspace(0, 5, 10)

plt.plot(xs, np.cos(xs))
ax = plt.gca()

axins = zoomed_inset_axes(ax, 2, loc="upper right")  # zoom = 6
axins.plot(xs, np.cos(xs))

# sub region of the original plot
x1, x2, y1, y2 = 1, 2, -0.25, 0.25
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.show()
