import itertools
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.markers as mark
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy


class ScatterData:
    colors = ["black", "green", "red"]
    markers = ["o", "v", "x"]

    def __init__(self, title, x_data):
        self.title = title
        self.x = x_data
        self.data = []
        self.limits = None, None

    def add_data(self, name, data, error=None):
        self.data.append((name, data, error))
        return self

    @property
    def size(self):
        return len(self.data)

    def x_lim(self, limits):
        self.limits = limits, self.limits[1]

    def y_lim(self, limits):
        self.limits = self.limits[0], limits

    def gen_colors(self):
        if len(self.data) <= len(self.colors):
            return self.colors[:len(self.data)]
        iterator = iter(cm.rainbow(numpy.linspace(0, 1, len(self.data))))
        return [next(iterator) for _ in range(len(self.data))]

    def gen_markers(self):
        if len(self.data) <= len(self.markers):
            return self.markers[:len(self.data)]
        iterator = itertools.cycle(mark.MarkerStyle.filled_markers)
        return [next(iterator) for _ in range(len(self.data))]

    def render(self, ax, lines=True, log_x=True, log_y=True, label_x=None, label_y=None, legend_pos="lower right",
               x_ticks=None):
        plots = []
        colors = self.gen_colors()
        markers = self.gen_markers()

        for i in range(self.size):
            title, times, error = self.data[i]
            x_data = self.x(times) if callable(self.x) else self.x
            plots.append(ax.scatter(x_data, times, color=colors[i], marker=markers[i], s=40))
            if lines:
                ax.plot(x_data, times, color=colors[i])
            if error is not None:
                ax.errorbar(x_data, times, error, linestyle='None', color=colors[i])

        ax.grid(True)
        if len(self.data) < 10:
            ax.legend(plots, (t[0] for t in self.data), loc=legend_pos)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        x_lim, y_lim = self.limits
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        if label_y is not None:
            ax.set_ylabel(label_y)
        if label_x is not None:
            ax.set_xlabel(label_x)

        if x_ticks is not None:
            ax.xaxis.set_ticks(x_ticks)

    def plot(self, filename=None, size=None, **kwargs):
        fig = plt.figure()
        if size is not None:
            fig.set_size_inches(*size)
        self.render(fig.gca(), **kwargs)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, format="pdf")


def plot(file, *args, **kwargs):
    fig = plt.figure()
    fig.set_size_inches(12, 12)

    subplots = len(args)
    cols = int(numpy.ceil(numpy.sqrt(subplots)))
    rows = int(numpy.ceil(subplots / cols))

    import matplotlib.gridspec as grid_spec
    gs = grid_spec.GridSpec(rows, cols)

    axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, :])]
    legend_positions = ["lower right", "upper right", "lower left"]

    for i in range(subplots):
        legend_pos = legend_positions[i]
        args[i].render(axes[i], legend_pos=legend_pos, **kwargs)

    if file is None:
        plt.show()
    else:
        plt.savefig(file, format="pdf")
