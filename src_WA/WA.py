"""
Uses attrs
Adv: validators as method decorators, mypy works
Dis: pylance needs extra annotations (until numpy 1.20?), converters as separate functions
Note: mypy undestands that input types are for converter, and output types are as hinted
Look into: cattrs, attrs-serde, attrs-stict
"""

import json
from scipy.optimize import curve_fit
import numpy as np
from pathlib import Path
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import attr
# import cattr
# from xarray import DataArray
from typing import Callable, Mapping, NamedTuple, Sequence, cast, Tuple, Dict, Union, List  # noqa: F401

from typing_extensions import TypedDict, Literal  # noqa: F401


def water_added_to_pp(water_added: Sequence[float], vessel_volume: int = 1000) -> np.ndarray:
    """convert list of amounts of water in mg to array of partial pressures for a given volume"""
    wa_array = np.array(water_added, dtype=float)
    pp: np.ndarray = 4 * (wa_array / vessel_volume)
    # print(pp)
    return pp


def predict_y(x: np.ndarray, m: float, k: float, n: float, j: float) -> np.ndarray:
    """Predict y values for a given x value dataset, using the parameters supplied
    Equation: y = mx / (k+x) + 1/[(n/jx) - 1/j]"""
    form_A = (x * m) / (k + x)
    form_B_inv = (n / (j * x)) - (1 / j)
    form_B = 1 / form_B_inv
    y_fit: np.ndarray = form_A + form_B
    return y_fit


def predict_y_maxed_humidity(x: np.ndarray, m: float, k: float, j: float) -> np.ndarray:
    """Predict y values for a given x value dataset, using the parameters supplied
    Equation: y = mx / (k+x) + 1/[(n/jx) - 1/j] where n = 1"""
    form_A = (x * m) / (k + x)
    form_B_inv = (1 / (j * x)) - (1 / j)
    form_B = 1 / form_B_inv
    # print(form_B)
    y_fit: np.ndarray = form_A + form_B
    return y_fit


class ParamsNTup(NamedTuple):
    """Container for parameters"""
    # used instead of dataclass as has .__iter__() and indexable
    m: float = 300
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
            "Fit parameters should be container of len == 4, eg. ParamsNTup")
    try:
        rounded_v = tuple((round(x, 2) for x in v))
        w = ParamsNTup(*rounded_v)
    except TypeError as terr:
        terr.args += ("Fit parameters should be a ParamsType (ParamsNTup or list | tuple | set | ndarray)",)
        raise
    return w


@ attr.dataclass()
class WaterAbsFitParams():
    """Holds parameters for fit equation: y = mx / (k+x) + 1/[(n/jx) - 1/j]
    attrs: .params
    methods: .as_tuple(), .as_dict(), __len__()"""
    params: ParamsNTup = attr.ib(ParamsNTup(), converter=convert_params)
    std_errs: ErrType = attr.ib((0, 0, 0, 0))

    @ params.validator
    def validate_params(self, attribute: attr.Attribute, v: ParamsNTup) -> None:
        if (
            not isinstance(v, ParamsNTup)
            or not isinstance(v[0], (int, float))
            or len(v) != 4
        ):
            raise TypeError(
                "Fit parameters should by a ParamsNTup (coerced from tuple, list, set, np.ndarray)")

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
        return ParamsTDict(m=d['m'], k=d['k'], n=d['n'], j=d['j'])


def get_params(x_data: np.ndarray, y_data: np.ndarray,
               init_params: Union[WaterAbsFitParams,
                                  ParamsType] = ParamsNTup(),
               fix_n: bool = False,
               ) -> WaterAbsFitParams:

    init_pt = ParamsNTup(*init_params) if not isinstance(init_params,
                                                         WaterAbsFitParams) else init_params.params

    assert len(x_data) == len(y_data)

    popt: np.ndarray
    pcov: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    pcov_diags: Tuple4float

    if fix_n:
        popt, pcov = curve_fit(predict_y_maxed_humidity, x_data, y_data,
                               p0=(init_pt.m, init_pt.k, init_pt.j),
                               bounds=([0, 0, 0], [np.inf, 1, np.inf])
                               )
        popt = np.insert(popt, 2, values=1, axis=0)
        pcov_diags = (pcov[0][0], pcov[1][1], 0, pcov[2][2])
    else:
        popt, pcov = curve_fit(predict_y, x_data, y_data,
                               p0=init_pt,
                               bounds=([0, 0, 0, 0], [
                                       np.inf, 1, np.inf, np.inf])
                               )
        pcov_diags = (pcov[0][0], pcov[1][1], pcov[2][2], pcov[3][3])
    assert len(popt) == len(init_pt) == 4
    std_errs = cast(ErrType, tuple(round(x**0.5, 2) for x in pcov_diags))
    return WaterAbsFitParams(popt, std_errs)


def wabs_plot(title: str = "") -> Tuple[Figure, Axes]:
    fig = plt.figure()
    ax: Axes = fig.subplots()   # type:ignore
    plt.xlim(0, 1)
    plt.ylim(0, 600)
    plt.ylabel("Mass Water Absorbed (ug / ug MOx)")
    plt.xlabel("P / P0")
    ax.set_title(title)
    return (fig, ax)


def plot_line(x_data: np.ndarray,
              y_data: np.ndarray,
              fig: Figure,
              ax: Axes,
              legend: str = "",
              *,
              linecolor: str = 'b',
              fix_n: bool = False,
              vmax_line: bool = False,
              ) -> Tuple[Figure, Axes]:
    """ b: blue
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
    fit_data = get_params(x_data, y_data, fix_n=fix_n)
    print(fit_data.params)
    x_fit = np.linspace(0.001, 0.99, 1000)
    y_fit = predict_y(x_fit, *fit_data.params)

    ax.plot(x_data[1:], y_data[1:], f'{linecolor}o', label=legend)
    ax.plot(x_fit, y_fit, f'{linecolor}-')
    if vmax_line:
        ax.hlines(fit_data.m, xmin=0, xmax=1,
                  colors=linecolor, linestyles='--')

    plt.legend(loc="upper left")
    return fig, ax


@attr.dataclass()
class WA_dataset():
    x_data: np.ndarray = attr.ib()
    y_data: np.ndarray = attr.ib()
    legend: str = attr.ib()
    color: str = attr.ib("")

    def __attrs_post_init__(self) -> None:
        """convert input arrays to floats and insert inital near 0 values"""
        # TODO: move to converter function / validator method
        self.x_data = np.insert(self.x_data.astype(float), 0, 0.001, axis=0)
        self.y_data = np.insert(self.y_data.astype(float), 0, 0.01, axis=0)
        assert len(self.x_data) == len(self.y_data)


@attr.dataclass()
class PlotInput():
    title: str
    data: List[WA_dataset]


class PlotInputTDict(TypedDict):
    title: str
    data: List[WA_dataset]


PlotInputMap = Mapping[str, Union[str, List[WA_dataset]]]


def plot_multi(input: Union[PlotInput, PlotInputTDict, PlotInputMap],
               vmax_line: bool = False,
               fix_n: bool = True,
               show: bool = True,
               save: bool = True,
               save_path: Union[str, Path] = ".",
               file_name: str = "plot_multi.png"
               ) -> Tuple[Figure, Axes]:
    """plot multiple Water Abs datasets on single axis
    vmax_line: bool -> add a line for v_max for each  dataset
    fix_n: bool = True -> Use constrained n(max partial pressure) of 1
    show: bool = True -> show figure
    save: bool = True -> save figure to provided path or in cwd
    save_path: str | Path = R".\" -> path to save figure, if save=True  """

    valid_input: PlotInput = (input if isinstance(input, PlotInput)
                              else PlotInput(**input))

    fig, ax = wabs_plot(title=valid_input.title)

    for item in valid_input.data:
        # assert isinstance(item, WA_dataset)
        plot_line(item.x_data, item.y_data,
                  fig, ax,
                  item.legend, linecolor=item.color,
                  vmax_line=vmax_line, fix_n=fix_n)

    if save:
        sp = Path(save_path) / file_name
        plt.savefig(f'{sp}')
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":

    # CeO2
    x_ceo2_ox = water_added_to_pp([10, 25, 50, 75, 100, 150, 200, 245])
    y_ceo2_ox = np.array([120, 200, 235, 300, 310, 320, 420, 550])

    x_ceo2_n = water_added_to_pp([10, 25, 50, 75, 100, 150, 200, 245])
    y_ceo2_n = np.array([110, 140, 205, 250, 280, 300, 320, 450])

    ex: PlotInputTDict = {
        'title': "Ceria oxalate vs nitrate",
        'data': [WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO2 (oxalate)", color='b'),
                 WA_dataset(x_ceo2_n, y_ceo2_n, "CeO2 (nitrate)", color='c'),
                 ]
    }

    ceria_plot_500 = plot_multi(input=ex)

    """
    # ThO2
    x_tho2 = np.array([0.1, 10, 25, 50, 75, 100, 150, 200, 245])
    y_tho2 = np.array([0.1, 120, 200, 235, 300, 310, 320, 420, 550])

    plot_line(x_tho2, y_tho2, "ThO2", linecolor='r')

    # ThO2
    x_tho2 = np.array([0.1, 10, 25, 50, 75, 100, 150, 200, 245])
    y_tho2 = np.array([0.1, 120, 200, 235, 300, 310, 320, 420, 550])

    plot_line(x_tho2, y_tho2, "ThO2", linecolor='r')
    """

    # make plot_line colors literals and autocycle colors
