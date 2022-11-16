import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

class RenderPlot:
    def __init__(self, metric_arr):
        self._plot_array = metric_arr
