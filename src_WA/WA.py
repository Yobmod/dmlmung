"""
Uses attrs
Adv: validators as method decorators, mypy works
Dis: pylance needs extra annotations (until numpy 1.20?), converters as separate functions
Note: mypy undestands that input types are for converter, and output types are as hinted
Look into: cattrs, attrs-serde, attrs-stict, related
"""

import json
from scipy.optimize import curve_fit
from scipy import stats
import numpy as np
from numpy import ndarray
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import attr
# import cattr
# from xarray import DataArray
# import pandas as pd
# from numpy import typing as npt
from typing import (Mapping, NamedTuple, Sequence,
                    Any, cast, Tuple, Dict, Union, List, )
from typing_extensions import TypedDict, Literal  # noqa: F401


def water_added_to_pp(water_added: Sequence[float], vessel_volume: int = 1000
                      ) -> 'np.ndarray[Any, float]':
    """convert list of amounts of water in mg to array of partial pressures for a given volume"""
    wadd_array = np.asarray(water_added, dtype=float)
    pp = np.multiply(wadd_array / vessel_volume, 4)
    assert isinstance(pp, ndarray)
    return pp


def freq_change_to_WA(freq: Sequence[float], mass_mat: float, F0: int,
                      elec_area: float = 0.66,
                      ) -> 'np.ndarray[Any, float]':
    """convert list of frequence changes in Hz to array of
    water absorbed per mass of absorbant (ug / ug)"""
    Pgxugroot = 875500
    n2F0F0: int = 2 * (F0 ** 2)
    cf = n2F0F0 / Pgxugroot

    freq_array = np.asarray(freq, dtype=float)
    wabs_array = ((freq_array / cf) * elec_area)  # water abs per electrode
    wabs_per_g = wabs_array / (mass_mat / 1_000_000)
    assert isinstance(wabs_per_g, ndarray)
    return wabs_per_g


def predict_y(x: np.ndarray, m: float, k: float, n: float, j: float) -> np.ndarray:
    """Predict y values for a given x value dataset, using the parameters supplied
    Equation: y = mx / (k+x) + 1/[(n/jx) - 1/j]"""
    form_A = (x * m) / (k + x)
    # form_B_inv = (n / (j * x)) - (1 / j)  # = jn/jjx -jx/jjx = n - x / jx
    form_B = (j * x) / (n - x)  # 1 / form_B_inv
    y_fit = form_A + form_B
    assert isinstance(y_fit, ndarray)
    return y_fit


def predict_y_maxed_humidity(x: np.ndarray, m: float, k: float, j: float) -> np.ndarray:
    """Predict y values for a given x value dataset, using the parameters supplied
    Equation: y = mx / (k+x) + 1/[(n/jx) - 1/j] where n = 1"""
    form_A = (x * m) / (k + x)
    # form_B_inv = (1.0 / (j * x)) - (1 / j)  # 1.0 is n
    form_B = (j * x) / (1.0 - x)  # 1 / form_B_inv
    y_fit = form_A + form_B
    assert isinstance(y_fit, ndarray)
    return y_fit


class ParamsNTup(NamedTuple):
    """Container for parameters"""

    # used instead of dataclass as has .__iter__() and indexable
    m: float = 0.03
    k: float = 0.5
    n: float = 1
    j: float = 3


class ParamsTDict(TypedDict):
    """TypedDict for parameters"""

    m: float
    k: float
    n: float
    j: float


CorrelationType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Tuple4float = Tuple[float, float, float, float]
Tuple3float = Tuple[float, float, float]
ErrType = Tuple4float
ParamsType = Union[List[float], Tuple4float, ParamsNTup, np.ndarray]


def convert_params(v: ParamsType) -> ParamsNTup:
    """Converter function to coerce 4 float list, tuple, set, ndarray to ParamsNTup
    Also rounds floats to 2 d.p."""
    if len(v) != 4:
        raise ValueError(
            "Fit parameters should be container of len == 4, eg. ParamsNTup"
        )
    try:
        rounded_v = tuple((round(x, 3) for x in v))
        w = ParamsNTup(*rounded_v)
    except TypeError as terr:
        terr.args += (
            "Fit parameters should be a ParamsType (ParamsNTup or list | tuple | set | ndarray)",
        )
        raise
    return w


@attr.dataclass()
class WaterAbsFitParams:
    """Holds parameters for fit equation: y = mx / (k+x) + 1/[(n/jx) - 1/j]
    attrs: .params
    methods: .as_tuple(), .as_dict(), __len__()"""

    params: ParamsNTup = attr.ib(ParamsNTup(), converter=convert_params)
    std_errs: ErrType = attr.ib((0, 0, 0, 0))

    @params.validator
    def validate_params(self, attribute: attr.Attribute, v: ParamsNTup) -> None:
        if (
            not isinstance(v, ParamsNTup)
            or not isinstance(v[0], (int, float))
            or len(v) != 4
        ):
            raise TypeError(
                "Fit parameters should by a ParamsNTup (coerced from tuple, list, set, np.ndarray)"
            )

        if any(p <= 0 for p in v):
            print(v)
            raise ValueError(
                "All fit parameters should be positive floats | ints")

    def __attrs_post_init__(self) -> None:
        self.m: float = self.params.m
        self.k: float = self.params.k
        self.n: float = self.params.n
        self.j: float = self.params.j

    @classmethod
    def __len__(cls) -> int:
        """use len() to get number of fields"""
        return len(attr.fields(cls))

    def as_tuple(self) -> Tuple[ParamsNTup, ErrType]:
        """return datclass as Tuple[ParamsNTup, ErrType]"""
        t = attr.astuple(self, recurse=False)

        return cast(Tuple[ParamsNTup, ErrType], t)

    def as_dict(self) -> Dict[str, Union[ParamsNTup, ErrType]]:
        """return datclass as Dict[str, Union[ParamsNTup, ErrType]]"""
        d: Dict[str, Union[ParamsNTup, ErrType]
                ] = attr.asdict(self, recurse=False)
        return d

    def as_json(self) -> str:
        """return datclass as string formatted as Dict[str, List[float]]"""
        d = attr.asdict(self, recurse=True)
        return json.dumps(d)

    def params_dict(self) -> ParamsTDict:
        d = self.params._asdict()
        return ParamsTDict(m=d["m"], k=d["k"], n=d["n"], j=d["j"])


def get_params(x_data: np.ndarray,
               y_data: np.ndarray,
               init_params: Union[WaterAbsFitParams, ParamsType],
               fix_n: bool = False,
               ) -> WaterAbsFitParams:

    init_pt = (ParamsNTup(*init_params)
               if not isinstance(init_params, WaterAbsFitParams)
               else init_params.params)

    assert len(x_data) == len(y_data)

    popt: np.ndarray
    pcov: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    pcov_diags: Tuple4float

    if fix_n:
        popt, pcov = curve_fit(predict_y_maxed_humidity, x_data, y_data,
                               p0=(init_pt.m, init_pt.k, init_pt.j),
                               bounds=([0, 0, 0], [np.inf, 1, np.inf]),
                               )
        popt = np.insert(popt, 2, values=1, axis=0)
        pcov_diags = (pcov[0][0], pcov[1][1], 0, pcov[2][2])
    else:
        popt, pcov = curve_fit(predict_y, x_data, y_data,
                               p0=init_pt,
                               bounds=([0, 0, 0, 0], [np.inf, 1, np.inf, np.inf]),
                               )
        pcov_diags = (pcov[0][0], pcov[1][1], pcov[2][2], pcov[3][3])
    assert len(popt) == len(init_pt) == 4
    std_errs = cast(ErrType, tuple(round(x ** 0.5, 3) for x in pcov_diags))
    return WaterAbsFitParams(popt, std_errs)


def wabs_plot(title: str = "") -> Tuple[Figure, Axes]:
    fig = plt.figure()
    ax: Axes = fig.subplots()  # type:ignore
    plt.xlim(0, 1)
    plt.ylim(0, 0.1)
    plt.ylabel(r"Mass Water Absorbed (g / g $\mathrm{MO_x}$")
    plt.xlabel("H\u2082O P / P\u2080")
    ax.set_title(title)
    return (fig, ax)


def plot_line(x_data: 'np.ndarray[Any, float]', y_data: np.ndarray,
              fig: Figure, ax: Axes,
              legend: str = "",
              *,
              linecolor: str = "b",
              fix_n: bool = True,
              vmax_line: bool = True,
              ) -> Tuple[Figure, Axes]:
    """b: blue
    g: green
    r: red
    c: cyan
    m: magenta
    y: yellow
    k: black
    w: white
    Tuple[float, float, float], Tuple4float
    html names"""

    assert len(x_data) == len(y_data)

    # get fit prameters, use to predict trendline
    fit_data = get_params(x_data, y_data, ParamsNTup(), fix_n=fix_n)
    print(fit_data.params)
    x_fit = np.linspace(0.001, 0.99, 1000)
    y_fit = predict_y(x_fit, *fit_data.params)

    ax.plot(x_data[1:], y_data[1:], f"{linecolor}o", label=legend)
    ax.plot(x_fit, y_fit, f"{linecolor}-")

    if vmax_line:
        # ax.plot(x_fit, np.repeat(fit_data.m, len(x_fit)), f"{linecolor}", linestyle="-")
        ax.hlines(fit_data.m, xmin=0, xmax=1.1, colors=linecolor, linestyles=(0, (5, 5)))
        # linestyle = "--" or ":" breaks curve_fit somehow

    plt.legend(loc="upper left")
    return fig, ax


@attr.dataclass()
class WA_dataset:
    x_data: np.ndarray = attr.ib()
    y_data: np.ndarray = attr.ib()
    legend: str = attr.ib()
    color: str = attr.ib("")

    def __attrs_post_init__(self) -> None:
        """convert input arrays to floats and insert initial near 0 values"""
        # TODO: move to converter function / validator method
        self.x_data = np.insert(self.x_data.astype(float), 0, 0.00001, axis=0)
        self.y_data = np.insert(self.y_data.astype(float), 0, 0.01, axis=0)
        assert len(self.x_data) == len(self.y_data)


@attr.dataclass()
class PlotInput:
    title: str = attr.ib()
    data: List[WA_dataset] = attr.ib()


class PlotInputTDict(TypedDict):
    title: str
    data: List[WA_dataset]


PlotInputMap = Mapping[str, Union[str, List[WA_dataset]]]


def plot_multi(input: Union[PlotInput, PlotInputTDict],  # PlotInputMap],
               equation: bool = True,
               fix_n: bool = True,
               vmax_line: bool = False,
               show: bool = True,
               save: bool = True,
               save_path: Union[str, Path] = ".",
               file_name: str = "plot_multi.png",
               ) -> Tuple[Figure, Axes]:
    R"""plot multiple Water Abs datasets on single axis
    vmax_line: bool -> add a line for v_max for each  dataset
    fix_n: bool = True -> Use constrained n(max partial pressure) of 1
    show: bool = True -> show figure
    save: bool = True -> save figure to provided path or in cwd
    save_path: str | Path = R".\" -> path to save figure, if save=True"""
    valid_input = (input
                   if isinstance(input, PlotInput)
                   else PlotInput(**input)
                   )

    fig, ax = wabs_plot(title=valid_input.title)

    for item in valid_input.data:
        # assert isinstance(item, WA_dataset)
        plot_line(item.x_data, item.y_data,
                  fig, ax,
                  item.legend,
                  linecolor=item.color,
                  vmax_line=vmax_line,
                  fix_n=fix_n,
                  )

    if equation:
        y_min, y_max = plt.ylim()
        plt.text(x=0.4, y=y_max * 0.9,  # m => a, k => b, j =? c, n => d
                 s=r"$ y =  \dfrac{a \cdot x}{b + x} + \dfrac{c \cdot x}{d - x}$",
                 fontsize=12,
                 )
        plt.text(x=0.4, y=y_max * 0.65,
                 s="a: max. water absorbed (---)\n"
                 + "b: P/P\u2080 @ a/2 \n"
                 + "c: water condensed @ b/2 (P/P\u2080 = 0.5) \n"
                 + "d: max. P/P\u2080 (P/P\u2080 = 1.0) \n",
                 fontsize=9,
                 )

    if save:
        sp = Path(save_path) / file_name
        plt.savefig(f"{sp}")

    if show:
        plt.show()

    return fig, ax


def trunc_data_for_bet(x_data: np.ndarray, y_data: np.ndarray, max_x: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    x_trunc: np.ndarray = np.asarray([x for x in x_data if x < max_x])
    print(x_trunc)
    y_trunc: np.ndarray = 1 / y_data[0:len(x_trunc)]
    print(y_trunc)
    return (x_trunc, y_trunc)


def get_bet_params(x_data: np.ndarray, y_data: np.ndarray, max_x: float = 0.4) -> Tuple[float, float]:
    x_trunc, y_trunc = trunc_data_for_bet(x_data, y_data, max_x=max_x)
    slope, intercept, r, p, stderr = stats.linregress(x_trunc, y_trunc)
    print(slope, intercept)
    return (slope, intercept)


def bet_plot(title: str = "") -> Tuple[Figure, Axes]:
    fig = plt.figure()
    ax: Axes = fig.subplots()  # type:ignore
    plt.xlim(0, 1)
    # plt.ylim(0, 0.1)
    plt.ylabel(r"BET Water Absorbtion (units $\mathrm{MO_x}$")
    plt.xlabel("H\u2082O P / P\u2080")
    ax.set_title(title)
    return (fig, ax)


def plot_bet_line(x_data: np.ndarray, y_data: np.ndarray,
                  fig: Figure, ax: Axes,
                  legend: str = "",
                  *,
                  linecolor: str = "b",
                  # fix_n: bool = True,
                  ) -> Tuple[Figure, Axes]:
    """b: blue
    g: green
    r: red
    c: cyan
    m: magenta
    y: yellow
    k: black
    w: white
    Tuple[float, float, float], Tuple4float
    html names"""

    assert len(x_data) == len(y_data)

    # fit to straight line for < 0.3
    slope, intercept = get_bet_params(x_data, y_data)
    x_fit = np.linspace(0.001, 0.99, 1000)
    y_fit = np.add(np.multiply(x_fit, slope), intercept)
    # assert isinstance(y_fit, ndarray)
    ax.plot(x_data[1:], 1 / y_data[1:], f"{linecolor}o", label=legend)
    ax.plot(x_fit, y_fit, f"{linecolor}-")

    plt.legend(loc="upper left")
    return fig, ax


def plot_bet_multi(input: Union[PlotInput, PlotInputTDict],  # PlotInputMap],
                   fix_n: bool = True,
                   show: bool = True,
                   save: bool = True,
                   save_path: Union[str, Path] = ".",
                   file_name: str = "plot_bet_multi.png",
                   ) -> Tuple[Figure, Axes]:
    R"""plot multiple Water Abs datasets as BET on single axis
    vmax_line: bool -> add a line for v_max for each  dataset
    fix_n: bool = True -> Use constrained n(max partial pressure) of 1
    show: bool = True -> show figure
    save: bool = True -> save figure to provided path or in cwd
    save_path: str | Path = R".\" -> path to save figure, if save=True"""
    valid_input: PlotInput = (input
                              if isinstance(input, PlotInput)
                              else PlotInput(**input)
                              )

    fig, ax = bet_plot(title=valid_input.title)

    for item in valid_input.data:
        # assert isinstance(item, WA_dataset)
        plot_bet_line(item.x_data, item.y_data,
                      fig, ax,
                      item.legend,
                      linecolor=item.color,
                      # fix_n=fix_n,
                      )

    if save:
        sp = Path(save_path) / file_name
        plt.savefig(f"{sp}")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":

    # ThO2
    x_ceo2_ox = water_added_to_pp(
        [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
    y_ceo2_ox = freq_change_to_WA(
        [146, 91, 198, 170, 224, 205, 288, 269, 295, 286, 312, 295, 374, 403, 502, 548],
        mass_mat=69, F0=5791070
    )

    x_tho2_ox = water_added_to_pp(
        [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
    y_tho2_ox = freq_change_to_WA(
        [180, 180, 211.5, 211, 275, 275, 305, 305, 311.5, 311, 333.5, 333, 431.5, 431, 529, 540],
        mass_mat=70.5, F0=5794505
    )

    oxal_data_500: PlotInputTDict = {
        "title": "Ceria vs Thoria (from oxalate)",
        "data": [
            WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO\u2082 (oxalate)", color="c"),
            WA_dataset(x_tho2_ox, y_tho2_ox, "ThO\u2082 (oxalate)", color="g"),
        ],
    }

    oxalate_plot_500 = plot_multi(input=oxal_data_500, file_name="plot_oxalate.png", fix_n=True, vmax_line=True)
    oxalate_bet_500 = plot_bet_multi(input=oxal_data_500, file_name="bet_oxalate.png")

    # make plot_line colors literals and autocycle colors
