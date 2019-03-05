import numpy as np

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

desc = np.asarray(MAP, dtype='c')
print(desc)