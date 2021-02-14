"""
Uses attrs
Adv: validators as method decorators, mypy works
Dis: pylance needs extra annotations (until numpy 1.20?), converters as separate functions
Note: mypy undestands that input types are for converter, and output types are as hinted
Look into: cattrs, attrs-serde, attrs-stict, related
"""

# TODO if save_path, call a function to check path exists or create

import csv
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
from typing import (Mapping, NamedTuple, Sequence,
                    Any, cast, Tuple, Dict, Union, List, )
from typing_extensions import TypedDict, Literal  # noqa: F401


def calc_water_saturation_pressure(temp: float) -> float:
    """ Calculate saturated vapour pressure for a given temperature using formula from: 
    https://journals.ametsoc.org/jamc/article/57/6/1265/68235/A-Simple-Accurate-Formula-for-Calculating
    """
    const_b = 4924.99 / (temp + 237.1)
    const_c = (temp + 105) ** 1.57
    p0: float = (np.exp(34.494 - const_b) / (const_c)) * 1.03
    return p0


def water_added_to_pp(water_added: Sequence[float],
                      vessel_volume: int = 1000,
                      temp: int = 75,
                      ) -> np.ndarray:
    """convert list of amounts of water in mg to array of partial pressures for a given volume"""
    p0 = calc_water_saturation_pressure(temp=temp)

    vol_m3 = vessel_volume / 1_000_000
    temp_K = temp + 273.15

    water_moles = np.asarray(water_added, dtype=float) / (18.02 * 1000)
    pp = (water_moles * 8.314 * temp_K) / (vol_m3 * p0)
    # print(pp)
    assert isinstance(pp, ndarray)
    return pp


def freq_change_to_WA(freq: Sequence[float], mass_mat: float, F0: int,
                      elec_area: float = 0.342,
                      ) -> np.ndarray:
    """convert list of frequence changes in Hz to array of
    water absorbed per mass of absorbant (ug / ug)"""
    Pgxugroot = 875500
    n2F0F0: int = 2 * (F0 ** 2)
    cf = n2F0F0 / Pgxugroot

    freq_array = np.asarray(freq, dtype=float)
    wabs_array = ((freq_array / cf) * elec_area)  # water abs per electrode
    # print(wabs_array)
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
    k: float = 0.05
    n: float = 1
    j: float = 0.1


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
        rounded_v = tuple((round(x, 5) for x in v))
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
            # raise ValueError("All fit parameters should be positive floats | ints")

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
                               # bounds=([0, 0, 0], [np.inf, 1, np.inf]),
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
    plt.ylim(0, 0.01)
    plt.ylabel(r"Mass Water Absorbed (g / g $\mathrm{MO_x}$)")
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
    mass: float = attr.ib()
    temp: float = attr.ib()
    color: str = attr.ib("")
    slope: float = 0.0
    intercept: float = 0.0
    Hads: float = 0.0
    SA: float = 0.0

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
    valid_input = (input if isinstance(input, PlotInput)
                   else PlotInput(**input))

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
    """."""
    x_trunc: np.ndarray = np.asarray([x for x in x_data if x < max_x])
    # print(x_trunc)
    y_trunc: np.ndarray = y_data[0:len(x_trunc)]
    # print(y_trunc)
    assert len(x_trunc) == len(y_trunc)
    return (x_trunc, y_trunc)


def y_data_to_bet(x_data: np.ndarray, y_data: np.ndarray, mass: float, temp: float) -> np.ndarray:
    """Convert y_data of mass water per mass MOx to P/V(P0-P) for BET plot y-axis"""
    p0 = calc_water_saturation_pressure(temp)
    pp = x_data * p0
    pp_div_p0_min_pp = pp / (p0 - pp)

    y_vol_m3 = (y_data * mass) / (1_000_000 * 1_000_000)
    y_bet: np.ndarray[float] = pp_div_p0_min_pp / (y_vol_m3)
    return y_bet


def get_bet_params(
        x_data: np.ndarray, y_data: np.ndarray,
        *,
        mass: float,
        temp: float,
        max_x: float = 0.4) -> Tuple[
        float, float]:
    """y_trunc :: mass water (ug) per g MOx """
    x_trunc, y_trunc = trunc_data_for_bet(x_data, y_data, max_x=max_x)
    y_bet = y_data_to_bet(x_trunc, y_trunc, mass=mass, temp=temp)

    slope, intercept, r, p, stderr = stats.linregress(x_trunc, y_bet)
    # print(slope, intercept)
    return (slope, intercept)


def bet_plot(title: str = "", y_max: float = 0) -> Tuple[Figure, Axes]:
    fig = plt.figure()
    ax: Axes = fig.subplots()  # type:ignore
    plt.xlim(0, 1)
    if y_max:
        plt.ylim(0, y_max)
    plt.ylabel("P / V(P\u2080-P) (m$\mathrm{^{-3}}$)")
    plt.xlabel("H\u2082O P / P\u2080")
    ax.set_title(title)
    return (fig, ax)


def plot_bet_line(x_data: np.ndarray, y_data: np.ndarray,
                  fig: Figure, ax: Axes,
                  legend: str = "",
                  *,
                  mass: float,
                  temp: float,
                  linecolor: str = "b",
                  equation: bool = True,
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

    # plot truncated data thats used for fit
    x_trunc, y_trunc = trunc_data_for_bet(x_data, y_data, max_x=0.4)
    y_bet = y_data_to_bet(x_trunc, y_trunc, mass=mass, temp=temp)
    ax.plot(x_trunc, y_bet, f"{linecolor}o", label=legend)

    # fit to straight line for < 0.4
    slope, intercept = get_bet_params(x_trunc, y_trunc, mass=mass, temp=temp)
    x_fit = np.linspace(0.001, 0.99, 1000)
    y_fit = np.add(np.multiply(x_fit, slope), intercept)
    ax.plot(x_fit, y_fit, f"{linecolor}-")

    # plot un-truncated data to show no fit
    y_none_fitted = y_data_to_bet(x_data, y_data, mass=mass, temp=temp)
    ax.plot(x_data, y_none_fitted, f"{linecolor}x")

    plt.legend(loc="upper left")

    return fig, ax


def plot_bet_multi(input: Union[PlotInput, PlotInputTDict],  # PlotInputMap],
                   show: bool = True,
                   save: bool = True,
                   save_path: Union[str, Path] = ".",
                   file_name: str = "plot_bet_multi.png",
                   y_max: float = 5e13,
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

    fig, ax = bet_plot(title=valid_input.title, y_max=y_max)

    for item in valid_input.data:
        # assert isinstance(item, WA_dataset)
        plot_bet_line(item.x_data, item.y_data,
                      fig, ax,
                      item.legend,
                      linecolor=item.color,
                      # fix_n=fix_n,
                      mass=item.mass,
                      temp=item.temp
                      )

    if save:
        sp = Path(save_path) / file_name
        plt.savefig(f"{sp}")

    if show:
        plt.show()

    return fig, ax


def add_results_to_WA_dataset(input: Union[PlotInput, PlotInputTDict],) -> PlotInput:
    valid_input: PlotInput = (input
                              if isinstance(input, PlotInput)
                              else PlotInput(**input)
                              )

    updated = []
    for item in valid_input.data:
        s, i = get_bet_params(item.x_data, item.y_data, mass=item.mass, temp=item.temp)
        Vm_m3 = (1 / (s + i))  # / 10
        c = (1 + (s / i))  # / 10
        temp_K = item.temp + 273
        Hads_kJ = ((np.log(c) * temp_K * 8.314) + 41770) / 1000
        SA_m2 = Vm_m3 * 6.02e23 * 1.60E-19 / (18.02 / 1_000_000)
        SA_per_g = SA_m2 / (item.mass / 1_000_000)
        item.slope = s
        item.intercept = i
        item.Hads = Hads_kJ
        item.SA = SA_per_g
        # print(Vm_m3, Hads_kJ, SA_m2, SA_per_g)
        updated.append(item)

    return PlotInput(title=valid_input.title, data=updated)


def table_bet_multi(input: Union[PlotInput, PlotInputTDict],
                    save: bool = True,
                    save_path: Union[str, Path] = ".",
                    file_name: str = "table_bet_multi.csv",
                    ) -> None:

    valid_input: PlotInput = (input
                              if isinstance(input, PlotInput)
                              else PlotInput(**input)
                              )

    if save:
        table_data = add_results_to_WA_dataset(valid_input)

        sp = Path(save_path) / file_name
        with open(sp, mode='w', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['Name        ', 'mass', 'temp', 'slope', 'interecept', 'Hads (kJ/mol)', 'SA(m3/g)'])
            for item in table_data.data:
                w.writerow([item.legend,
                            item.mass,
                            item.temp,
                            f"{item.slope: .2f}",
                            f"{item.intercept: .2f}",
                            f"{item.Hads: .2f}",
                            f"{item.SA: .2f}",
                            ])


if __name__ == "__main__":

    x_ceo2_n = water_added_to_pp(
        [10, 10, 10,
         25, 25,
         50, 50, 50,
         75, 75,
         100, 100, 100,
         150, 150, 150,
         200, 200, 200,
         240, 240])
    y_ceo2_n = freq_change_to_WA(
        [1 * x for x in [
            10.4, 7.8, 9.9,
            19.0, 22.2,
            23.5, 25.9, 22.4,
            24.6, 28.0,
            33.4, 36.1, 24.6,
            35.8, 36.0, 25.6,
            47.4, 55.7, 32.8,
            80.0, 131.2]],
        mass_mat=46, F0=5751200
    )

    x_tho2_ox = water_added_to_pp(
        [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
    y_tho2_ox = freq_change_to_WA(
        [2 * x for x in [17.0, 11.2,
                         20.15, 23.1,
                         26.5, 32.5,
                         29.5, 30.2, 34.2, 33.4, 38.3, 41.1, 63.1, 52.5, 82.9, 156.0]],
        mass_mat=70.5, F0=5794505
    )

    x_uo2_ox = water_added_to_pp(
        [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
    y_uo2_ox = freq_change_to_WA(
        [2 * x for x in [8.1, 6.4,
                         11.9, 12.3,
                         16.6, 14.2,
                         20.8, 17.9,
                         28.1, 22.7,
                         34.2, 37.5,
                         45.4, 39.3,
                         74.2, 111.5]],
        mass_mat=55, F0=5798366)

    x_uo2_n = water_added_to_pp(
        [10, 20, 25, 25, 50, 50, 100, 150, 240])
    y_uo2_n = freq_change_to_WA(
        [2 * x for x in [4.5,
                         6.8,
                         12.5, 10.1,
                         16.3, 17.3,
                         20.8,
                         25.1,
                         100,
                         ]],
        mass_mat=65, F0=5763900
    )

    x_ceo2_ox = water_added_to_pp(
        [10, 10, 10, 10,
         25, 25, 25, 25,
         50, 50, 50, 50,
         75, 75, 75, 75,
         100, 100, 100, 100,
         150, 150, 150, 150,
         200, 200, 200, 200,
         240, 240, 240, 240
         ])
    y_ceo2_ox = freq_change_to_WA(
        [2 * x for x in [14.6, 11.1,  16.6, 9.1,
                         19.8, 16.3,  21.8, 19.0,
                         23.4, 21.5, 24.4, 22.5,
                         28.8, 27.9, 30.8, 28.9,
                         29.5, 36.6, 31.5, 30.6,
                         43.2, 39.5, 32.2, 30.5,
                         57.4, 61.3, 41.4, 44.3,
                         90.2, 154.8, 52.2, 56.8]],
        mass_mat=69, F0=5791070
    )

    data_500: PlotInputTDict = {
        "title": "Ceria vs Thoria vs Urania",
        "data": [
            WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO\u2082 (oxalate)", mass=69.0, temp=75, color="c"),
            WA_dataset(x_ceo2_n, y_ceo2_n, "CeO\u2082 (nitrate)", mass=69.0, temp=75, color="b"),
            WA_dataset(x_uo2_ox, y_uo2_ox, "UO\u2082 (oxalate)", mass=69.0, temp=75, color="m"),
            WA_dataset(x_uo2_n, y_uo2_n, "UO\u2082 (nitrate)", mass=69.0, temp=75, color="r"),
            WA_dataset(x_tho2_ox, y_tho2_ox, "ThO\u2082 (oxalate)", mass=70.5, temp=75, color="g"),
        ],
    }

    plot_500 = plot_multi(input=data_500,
                          file_name="plot_fixed_p.png",
                          fix_n=True, vmax_line=False, equation=True)
    bet_500 = plot_bet_multi(input=data_500, file_name="bet_oxalate.png", y_max=2e13)
    table_bet_multi(data_500, file_name="bet_oxalate.tsv")
