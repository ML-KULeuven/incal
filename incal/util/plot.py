import itertools
import platform

import matplotlib as mpl

if platform.system() == "Darwin":
    mpl.use('TkAgg')

import matplotlib.markers as mark
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy


class ScatterData:
    colors = ["black", "green", "red"]
    markers = ["o", "v", "x"]

    def __init__(self, title, plot_options):
        self.title = title
        self.data = []
        self.limits = None, None
        self.plot_options = plot_options

    def add_data(self, name, x_data, y_data, error=None):
        self.data.append((name, x_data, y_data, error))
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

    def render(self, ax, lines=True, log_x=True, log_y=True, label_x=None, label_y=None, legend_pos=None,
               x_ticks=None):

        plots = []
        colors = self.gen_colors()
        markers = self.gen_markers()

        if legend_pos is None:
            legend_pos = "lower right"

        plot_diagonal = False
        plot_extra = None
        plot_format = "scatter"
        show_error = True
        steps_x = None

        cache = None
        for plot_option in self.plot_options or ():
            if cache is None:
                if plot_option == "diagonal":
                    plot_diagonal = True
                else:
                    cache = plot_option
            else:
                if cache == "format":
                    plot_format = plot_option
                elif cache == "error":
                    show_error = (plot_option == 1)
                elif cache == "legend_pos":
                    legend_pos = plot_option
                elif cache == "lx":
                    label_x = plot_option
                elif cache == "ly":
                    label_y = plot_option
                elif cache == "steps_x":
                    steps_x = int(plot_option)
                elif cache == "plot_extra":
                    plot_extra = plot_option
                cache = None

        min_x, max_x, min_y, max_y = numpy.infty, -numpy.infty, numpy.infty, -numpy.infty
        for i in range(self.size):
            name, x_data, y_data, error = self.data[i]
            try:
                min_x = min(min_x, numpy.min(x_data))
                min_y = min(min_y, numpy.min(y_data))
                max_x = max(max_x, numpy.max(x_data))
                max_y = max(max_y, numpy.max(y_data))
            except TypeError:
                pass

            if plot_format == "scatter":
                plots.append(ax.scatter(x_data, y_data, color=colors[i], marker=markers[i], s=40))
                if lines:
                    ax.plot(x_data, y_data, color=colors[i])
                if show_error and error is not None:
                    ax.fill_between(x_data, y_data - error, y_data + error, color=colors[i], alpha=0.35,
                                    linewidth=0)
                        # ax.errorbar(x_data, y_data, error, linestyle='None', color=colors[i])
            elif plot_format == "bar":
                plots.append(ax.bar(x_data, y_data, color=colors[i]))
            else:
                raise ValueError("Unknown plot format")

        if plot_diagonal:
            ax.plot(numpy.array([min_x, max_x]), numpy.array([min_y, max_y]), linestyle="--")
        if plot_extra and plot_extra == "1/x":
            ax.plot(x_data, 1 / x_data, linestyle="--")

        ax.grid(True)
        legend_names = list(t[0] for t in self.data)
        # legend_names = ["No mixing - DT", "No mixing - RF", "Mixing - DT", "Mixing - RF"]
        # legend_names = ["No formulas", "Formulas"]
        # legend_names = []
        if 10 > len(self.data) == len(legend_names):
            ax.legend(plots, legend_names, loc=legend_pos)

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

        # ax.set_ylim((0, 2))

        if steps_x is not None:
            x_ticks = numpy.linspace(min_x, max_x, steps_x)
        # x_ticks = [1, 2, 3]
        if x_ticks is not None:
            ax.xaxis.set_ticks(x_ticks)

    def plot(self, filename=None, size=None, **kwargs):
        fig = plt.figure()
        if size is not None:
            fig.set_size_inches(*size)
        self.render(fig.gca(), **kwargs)
        if filename is None:
            plt.show(block=True)
        else:
            plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0.08, dpi=600)


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
