import matplotlib.pyplot as plt
import numpy as np


x = {"a":1, "b" :3, "c" :1}
plt.bar(list(x.keys()), x.values(), color='g')
plt.show()