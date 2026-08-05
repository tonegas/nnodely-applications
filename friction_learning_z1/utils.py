"""Matplotlib utilities for consistent LaTeX plots."""

GOLDEN_RATIO: float = (5**0.5 - 1) / 2
DOCUMENT_WIDTH: float = 505.89

NN_COLORS = {
    "nnGreenLightest": "#D9FEEA",
    "nnGreenLight": "#A5FDCE",
    "nnGreenBase": "#51FCA0",
    "nnGreenDeep": "#45D38A",
    "nnGreenDeepest": "#39A572",

    "nnYellowLightest": "#FEF9D9",
    "nnYellowLight": "#FDF0A5",
    "nnYellowBase": "#FCE351",
    "nnYellowDeep": "#D2BE49",
    "nnYellowDeepest": "#A39641",

    "nnBlueLightest": "#DCD9FE",
    "nnBlueLight": "#ACA5FD",
    "nnBlueBase": "#5F51FC",
    "nnBlueDeep": "#5147D6",
    "nnBlueDeepest": "#413BAB",

    "nnRedLightest": "#FEDCD9",
    "nnRedLight": "#FDADA5",
    "nnRedBase": "#FC6151",
    "nnRedDeep": "#D25449",
    "nnRedDeepest": "#A34541",

    "nnPurpleLightest": "#FDD9FE",
    "nnPurpleLight": "#F9A5FD",
    "nnPurpleBase": "#F451FC",
    "nnPurpleDeep": "#CB47D6",
    "nnPurpleDeepest": "#9E3BAB",

    "nnCyanLightest": "#D9F6FF",
    "nnCyanLight": "#A5EAFF",
    "nnCyanBase": "#52D7FF",
    "nnCyanDeep": "#46B5D8",
    "nnCyanDeepest": "#398EAD",

    "nnGrayLightest": "#F9FAFB",
    "nnGrayLight": "#E5E7EB",
    "nnGrayBase": "#9CA3AF",
    "nnGrayDeep": "#4B5563",
    "nnGrayDeepest": "#111827",
}

NN_LOGO = {
    "nnLogo1": "#87D0FF",
    "nnLogo2": "#7DB0FE",
    "nnLogo3": "#7390FD",
    "nnLogo4": "#6970FC",
    "nnLogo5": "#5F51FC",
}

NN_CYCLE_BASE = [
    NN_COLORS["nnBlueBase"],
    NN_COLORS["nnGreenBase"],
    NN_COLORS["nnYellowBase"],
    NN_COLORS["nnRedBase"],
    NN_COLORS["nnPurpleBase"],
    NN_COLORS["nnCyanBase"],
]

NN_CYCLE_DEEP = [
    NN_COLORS["nnBlueDeep"],
    NN_COLORS["nnGreenDeep"],
    NN_COLORS["nnYellowDeep"],
    NN_COLORS["nnRedDeep"],
    NN_COLORS["nnPurpleDeep"],
    NN_COLORS["nnCyanDeep"],
]

def pt_to_inch(pt: float) -> float:
    """Convert pt to inches

    :param pt: The pt value to convert
    :return: The converted inches value
    """

    return pt / 72.27


def set_size(
    fraction: float = 1.0, subplots: tuple[int, int] = (1, 1)
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    :param fraction: Fraction of the width which you wish the figure to occupy
    :param subplots: The number of rows and columns of subplots
    :return: Dimensions of figure in inches, [width, height]
    """

    fig_width_in = pt_to_inch(DOCUMENT_WIDTH * fraction)
    fig_height_in = fig_width_in * GOLDEN_RATIO * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def get_tex_fonts() -> dict[str, bool | str | int]:
    """Get the font specs to comply with LaTeX

    :return: LaTeX font options
    """

    return {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 10,
    }

def get_palette(use_deep_cycle: bool = False) -> dict[str, str]:
    from cycler import cycler

    colors = NN_CYCLE_DEEP if use_deep_cycle else NN_CYCLE_BASE
    params = {
        "axes.prop_cycle": cycler(color=colors),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": NN_COLORS["nnGrayLight"],
        "axes.labelcolor": NN_COLORS["nnGrayDeepest"],
        "axes.titlecolor": NN_COLORS["nnGrayDeepest"],
        "xtick.color": NN_COLORS["nnGrayDeep"],
        "ytick.color": NN_COLORS["nnGrayDeep"],
        "grid.color": NN_COLORS["nnGrayLight"],
        "grid.linewidth": 1.0,
        "axes.grid": True,
        "legend.frameon": False,
    }
    return params

def get_plot_params() -> dict[str, bool | str | int | float]:
    """Get the plot params to comply with LaTeX

    :return: Plot parameters
    """

    return {
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "figure.constrained_layout.use": True,
        # "figure.dpi": 300,
    }


def get_style() -> str:
    """Get the style for the plots

    :return: The style of the plots
    """

    return "seaborn-v0_8-paper"


def make_label(symbol: str, unit: str) -> str:
    """Return a LaTeX axis label string like r"$v_x$ [m/s]".

    :param symbol: The symbol of the label
    :param unit: The unit of the symbol
    :return: The LaTeX axis label
    """

    return rf"${symbol}$ [{unit}]"


def linewidth_from_size(
    fig_width_in: float, base_width: float = 6.0, base_lw: float = 1.0
) -> float:
    """Scale linewidth relative to figure width.

    :param fig_width_in: Figure width in inches
    :param base_width: Reference width in inches
    :param base_lw: Linewidth at the reference width
    :return: Scaled linewidth
    """

    return base_lw * (fig_width_in / base_width)