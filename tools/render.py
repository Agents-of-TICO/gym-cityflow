import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

class RenderPlot:
    def __init__(self, metric_array, plt_title, plt_xlabel, plt_ylabel, line_color):
        self._plot_array = metric_array
        self.title = str(plt_title)
        self.xlab = str(plt_xlabel)
        self.ylab = str(plt_ylabel)
        self.color = str(line_color)

    def export_plot(self, file_name):
        fig = plt.figure()
        graph = fig.add_subplot()
        graph.plot(self._plot_array, color = self.color)
        plt.title(self.title)
        plt.xlabel(self.xlab)
        plt.ylabel(self.ylab)
        plt.savefig(file_name, format='png')
        graph.lines[0].remove()