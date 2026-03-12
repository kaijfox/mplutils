import seaborn as sns
import matplotlib.pyplot as plt
from typing import Protocol, Tuple, Callable, ClassVar, Optional

try:
    from typing_extensions import Self
except:
    from typing import Self
from pathlib import Path
import numpy as np
import logging

try:
    from IPython.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def palgen(name, skip=None):
    if isinstance(name, str):
        f = lambda n, **kw: sns.color_palette(name, n, **kw)
    else:
        f = name

    def make(n=None, **kw):
        if kw.get("as_cmap", False):
            return f(n, **kw)
        if skip is not None:
            if skip[1] is None:
                skip_end = n + skip[0]
                skip_add = skip[0]
            else:
                skip_end = -skip[1]
                skip_add = sum(skip)
            return f(n + skip_add, **kw)[skip[0] : skip_end]
        return f(n, **kw)

    return make


class colorset(Protocol):
    neutral: any
    subtle: any
    seq: Callable[[int], np.array]
    cts: Callable[[int], np.array]
    cts1: Callable[[int], np.array]
    C: np.array

    @staticmethod
    def make(name):
        return palgen(
            name,
        )


class vscode_dark_colors(colorset):
    neutral = ".8"
    subtle = ".4"
    seq = palgen("Spectral")
    cts = palgen("Blues_r")
    cts1 = palgen("viridis", skip=(2, None))
    C = ["C0", "C2", "C3", "C4"]
    rc = {}


class light_colors(colorset):
    neutral = ".1"
    subtle = ".8"
    seq = lambda n: sns.hls_palette(l=0.4, n_colors=n + 2)[1:-1]
    cts = palgen("magma")
    cts1 = palgen("viridis")
    C = ["C3", "C4", "C2"]
    rc = {}


class simple(light_colors):
    rc = {
        # font/text
        "font.family": "Arial, sans-serif",
        "font.size": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "mathtext.fontset": "cm",
        # axes
        "axes.facecolor": "ffffff00",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        # 'xtick.direction': 'in',
        # 'ytick.direction': 'in',
        # 'axes.autolimit_mode': 'round_numbers',
        # legend
        "legend.frameon": False,
        # lines/points
        "lines.markeredgewidth": 0,
        "lines.markersize": 4,
        "lines.linewidth": 1,
        "lines.color": "k",
    }


class compact(light_colors):
    rc = {
        # font/text
        "font.family": "Arial, sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "mathtext.fontset": "cm",
        # axes
        "axes.facecolor": "ffffff00",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        # 'xtick.direction': 'in',
        # 'ytick.direction': 'in',
        # 'axes.autolimit_mode': 'round_numbers',
        # legend
        "legend.frameon": False,
        # lines/points
        "lines.markeredgewidth": 0,
        "lines.markersize": 4,
        "lines.linewidth": 1,
        "lines.color": "k",
    }


color_sets = {
    "vscode_dark": vscode_dark_colors,
    "default": light_colors,
    "simple": simple,
    "compact": compact,
}
colorset.active = light_colors


def _display_path_with_copy(paths):
    """Display file path(s) with copy-to-clipboard buttons in Jupyter notebooks.

    Args:
        paths: Single path string or list of path strings
    """
    # Normalize to list
    path_list = paths if isinstance(paths, list) else [paths]

    if not HAS_IPYTHON:
        for p in path_list:
            print(str(Path(p).resolve()))
        return

    # Build HTML for all paths
    html_parts = ['<div style="font-family: monospace; margin: 5px 0;">']

    for path in path_list:
        abs_path = str(Path(path).resolve())
        html_parts.append(
            f"""
        <div style="margin: 2px 0;">
            <span>{path}</span>
            <a onclick="navigator.clipboard.writeText('{abs_path}').then(() => {{
                this.textContent = 'Copied!';
                setTimeout(() => {{ this.textContent = 'Copy'; }}, 1500);
            }})" style="margin-left: 10px; cursor: pointer;">Copy</a>
        </div>
        """
        )

    html_parts.append("</div>")
    display(HTML("".join(html_parts)))


class plot_finalizer(object):
    singleton: ClassVar[Optional[Self]] = None

    def __init__(self, plot_dir, **kws):
        self.plot_dir = plot_dir
        kws = {"fmt": "png", "save": True, **kws}
        self.fmt = kws.pop("fmt")
        self.save = kws.pop("save")
        self.display = kws.pop("display", True)
        self.despine = kws.pop("despine", True)
        self.allow_copy = kws.pop("allow_copy", False)
        self.kws = {"dpi": 300, "bbox_inches": "tight", **kws}
        plot_finalizer.singleton = self

    def finalize(
        self,
        fig=None,
        name=None,
        display=None,
        tight=True,
        despine=None,
        save=True,
        fmt=None,
        path=None,
        allow_copy=None,
        **kw,
    ):
        if fig is None:
            fig = plt.gcf()

        if despine is True or self.despine:
            for ax in fig.get_axes():
                sns.despine(ax=ax)
        if tight:
            fig.tight_layout()

        if (name is not None) and (save and self.save or save == "force"):
            plot_dir = self.plot_dir if path is None else path
            if not Path(plot_dir).exists():
                logging.warn(f"Creating plot directory: {plot_dir}")
                Path(plot_dir).mkdir(parents=True, exist_ok=True)

            # Handle fmt as list or single value
            fmt = self.fmt if fmt is None else fmt
            formats = fmt if isinstance(fmt, list) else [fmt]

            # Save all formats and collect paths
            out_files = []
            for format_type in formats:
                out_file = str(plot_dir) + "/" + name + "." + format_type
                out_files.append(out_file)
                fig.savefig(out_file, **self.kws, **kw)

            # Display all paths once
            if allow_copy is True or (allow_copy is None and self.allow_copy):
                _display_path_with_copy(out_files)
            else:
                for out_file in out_files:
                    print(out_file)

        if display is True or (self.display and display is not False):
            plt.show(fig)


def init_plt(plot_dir, style="vscode_dark", **kws) -> Tuple[colorset, plot_finalizer]:
    clr = color_sets[style]
    if len(clr.rc):
        plt.style.use("default")
        plt.rcParams.update(clr.rc)
    else:
        plt.style.use(style)
    sns.set_context("paper")
    colorset.active = clr
    plotter = plot_finalizer(plot_dir, **kws)
    return clr, plotter
