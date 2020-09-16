"""
Uses pydantic
Adv: validators as method decorators, mypy largely works
Dis: pylance doesnt work, no converters (use a _dummy field and post_init)
"""

from pydantic import validator
from scipy.optimize import curve_fit
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import astuple, asdict
from pydantic.dataclasses import dataclass
from typing import Callable, NamedTuple, cast, Tuple, Dict, Union, List

from typing_extensions import TypedDict


def predict_y(x: np.ndarray, m: float, k: float, n: float, j: float) -> np.ndarray:
    """y = mx / (k+x) + 1/[(n/jx) - 1/j]"""
    form_A = (x * m) / (k + x)
    form_B_inv = (n / (j * x)) - (1 / j)
    form_B = 1 / form_B_inv
    y_fit = form_A + form_B
    return y_fit


class ParamsNTup(NamedTuple):
    """Container for parameters"""
    # used instead of dataclass as has .__iter__() and indexable
    m: float
    k: float
    n: float
    j: float


class ParamsTDict(TypedDict):
    """TypedDict for parameters"""
    m: float
    k: float
    n: float
    j: float


CorrelationType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Tuple4float = Tuple[float, float, float, float]
ErrType = Tuple4float
ParamsType = Union[List[float], Tuple4float, ParamsNTup]


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class WaterAbsFitParams():
    """Holds parameters for fit equation: y = mx / (k+x) + 1/[(n/jx) - 1/j]
    attrs: .params
    methods: .as_tuple(), .as_dict(), __len__()"""
    init_params: ParamsType = ParamsNTup(300.0, 20.0, 250.0, 40.0)
    std_errs: ErrType = (0, 0, 0, 0)

    @validator('init_params', pre=True)
    def validate_params(cls, v: ParamsType) -> ParamsType:
        if not isinstance(v, (ParamsNTup, tuple, np.ndarray, list)) or not isinstance(v[0], (int, float)):
            raise TypeError(
                "Fit parameters should by a ParamsNTup (coerced from tuple, list, set, np.ndarray)")

        if not len(v) == 4:
            raise ValueError(
                "Fit parameters should be container of len == 4, eg. ParamsNTup")

        if not all(p > 0 for p in v):
            raise ValueError(
                "All fit parameters should be positive floats | ints")
        return v

    def convert_params(self) -> ParamsNTup:
        """Converter function to coerce 4 float list, tuple, set, ndarray to ParamsNTup
        Also rounds floats to 1 d.p. pydantic and coerces int to float"""
        v = self.init_params
        try:
            rounded_v = tuple((round(x, 1) for x in v))
            w = ParamsNTup(*rounded_v)
        except TypeError as terr:
            terr.args += ("Fit parameters should be a ParamsType (ParamsNTup or list | tuple | set | ndarray of floats)",)
            raise
        except Exception:
            raise
        return w

    def __post_init_post_parse__(self) -> None:
        self.params = self.convert_params()
        self.m: float = self.params.m
        self.k: float = self.params.k
        self.n: float = self.params.m
        self.j: float = self.params.j

    def __len__(self) -> int:
        """use len() to get number of fields"""
        return len(self.__dataclass_fields__)  # type:ignore

    def as_tuple(self) -> Tuple[ParamsNTup, ErrType]:
        """return datclass as Tuple[ParamsNTup, ErrType]"""
        return cast(Tuple[ParamsNTup, ErrType], astuple(self))

    def as_dict(self) -> Dict[str, Union[ParamsNTup, ErrType]]:
        """return datclass as Dict[str, Union[ParamsNTup, ErrType]]"""
        d: Dict[str, Union[ParamsNTup, ErrType]] = asdict(self)
        return d

    def params_dict(self) -> ParamsTDict:
        d = self.params._asdict()
        pd = ParamsTDict(m=d['m'], k=d['k'], n=d['n'], j=d['j'])
        return pd


def get_params(x: np.ndarray, y: np.ndarray,
               func: Callable = predict_y,
               init_params: Union[WaterAbsFitParams, ParamsType] = WaterAbsFitParams(),  # noqa: B008
               ) -> WaterAbsFitParams:

    init_params_tuple = init_params.params if isinstance(
        init_params, WaterAbsFitParams) else ParamsNTup(*init_params)

    assert len(x_data) == len(y_data)

    popt: np.ndarray
    pcov: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    popt, pcov = curve_fit(func, x_data, y_data,
                           p0=init_params_tuple,
                           bounds=([1, 1, 1, 1],
                                   [np.inf, np.inf, np.inf, np.inf]),
                           )
    assert len(popt) == len(init_params_tuple) == 4
    pcov_diags = (pcov[0][0], pcov[1][1], pcov[2][2], pcov[3][3])
    std_errs = cast(ErrType, tuple(round(x**0.5, 1) for x in pcov_diags))
    return WaterAbsFitParams(popt, std_errs)


x_data = np.array([0.1, 10, 25, 50, 75, 100, 150, 200, 245])
y_data = np.array([0.1, 120, 200, 235, 300, 310, 320, 420, 550])
assert len(x_data) == len(y_data)


# fit
fit_data = get_params(x_data, y_data)

# mock data
x_fit = np.linspace(1, 240, 50)
y_fit = predict_y(x_fit, *fit_data.params)

# plot
fig = plt.figure()
ax = fig.subplots()

ax.plot(x_data, y_data, 'r',
        label="real")
ax.plot(x_fit, y_fit, 'b',
        label="equation")
ax.plot(x_data, predict_y(x_data, *fit_data.params), 'g--',
        label='fit: a={}, b={}, c={}, d={}'.format(*fit_data.params))

plt.xlim(0, 250)
plt.ylim(0, 600)
plt.xlabel("x axis")
plt.ylabel("y axis")
ax.set_title('Simple plot title')
plt.legend(loc="upper left")
plt.show()

print(fit_data.params)
print(fit_data.params_dict())
print(
    f"testing: {len(fit_data.params)} params: {fit_data.m}, {fit_data.params_dict()['k']}, {fit_data.params.n}")
# print(fit_data.as_tuple())

t = WaterAbsFitParams((1, 2, 3, 4))
dic = t.as_dict()
a = len(t.as_dict())
at = t.as_tuple()
print(at)
print(t.params_dict())
