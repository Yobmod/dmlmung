# import json
from math import sqrt
from impedance.visualization import plot_nyquist
from impedance.models.circuits import CustomCircuit
from pathlib import Path
from my_types import paramsTup
from plot import (multi_bode_axes, multi_bode_add_data, multi_bode_plot_save,
                  get_json_settings,
                  multi_nyquist_add_data, multi_nyquist_axes, multi_nyquist_plot_save)

from mung import open_file_numpy, get_params, get_data_numpy, write_imp_data, write_zview_data
import snoop
import pprint
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Sequence
from dataclasses import dataclass

DEBUG = False
snoop.install(enabled=DEBUG)


@snoop(depth=1)
def main() -> None:
    data_dir = str(Path.cwd())
    settings = get_json_settings('settings')

    both_plot_exists = False
    phase_plot_exists = False
    gain_plot_exists = False
    nyquist_exists = False
    voltages = []

    for filepath in Path(data_dir).glob('*.csv'):

        filename = filepath.name
        params = get_params(data_dir, filename)  # get parameters from file name
        pprint.pprint(params.voltage)
        df = open_file_numpy(data_dir, filename)  # open file if csv and read into numpy array
        if df is not None and df.any():
            data = get_data_numpy(df)  # return impedance (5 columns) or cv data (4 columns)
        else:
            data = None
        if data and len(data) == 5:
            write_imp_data(data, params)  # writes to csv
            write_zview_data(data, params)
            (freq_log, imped_log, phase, imag_imped, real_imped) = data

        if not both_plot_exists:
            bode_both = multi_bode_axes(settings, y_axes="both")  # create axes
            both_plot_exists = True  # make truthy
        multi_bode_add_data(bode_both, freq_log, imped_log, phase, params)

        if not phase_plot_exists:
            bode_phase = multi_bode_axes(settings, y_axes="phase")  # create axes
            phase_plot_exists = True  # make truthy
            # multi_bode_title(bode_phase, params, settings)
        multi_bode_add_data(bode_phase, freq_log, imped_log, phase, params)

        if not gain_plot_exists:
            bode_gain = multi_bode_axes(settings, y_axes="gain")  # create axes
            gain_plot_exists = True  # make truthy
        multi_bode_add_data(bode_gain, freq_log, imped_log, phase, params)

        if not nyquist_exists:
            nyquist = multi_nyquist_axes(settings)
            nyquist_exists = True
        multi_nyquist_add_data(nyquist, imag_imped, real_imped, params)

        voltages.append(params.voltage)

        multi_bode_plot_save(bode_both, params)
        multi_bode_plot_save(bode_phase, params)
        multi_bode_plot_save(bode_gain, params)
        multi_nyquist_plot_save(nyquist, params)  # , set_colors='twilight',)

    # print(voltages)

    # plt.show()


@dataclass
class SingleData():
    params: paramsTup
    data: Tuple

    def __post_init__(self) -> None:
        self.freq_log, self.imped_log, self.phase, self.imag_imped, self.real_imped = self.data
        self.imped = self.Z = np.array(self.real_imped + 1j * -self.imag_imped, dtype=np.complex)[:, 0]
        self.freq = np.power(10, self.freq_log)[:, 0]
        self.voltage = self.params.voltage


@dataclass
class FittedSingleData():
    stored_data: SingleData
    circuit: str
    fit_params: Tuple
    chi2: float  # calc in DC?

    Z_fit: Sequence
    # Re_Z_fit: Sequence  # calc in DC?

    def __post_init__(self) -> None:
        self.voltage = self.stored_data.voltage
        self.C1 = self.fit_params[2]
        self.C2 = self.fit_params[5]


@dataclass
class FittedMultiData():
    data: Sequence[FittedSingleData]

    def __post_init__(self) -> None:
        self.voltages = [d.voltage for d in self.data]
        self.invsqc1s = [1 / (d.C1 ** 2 * 1000000000000) for d in self.data]
        self.invsqc2s = [1 / (d.C2 ** 2 * 1000000000000) for d in self.data]

    def split_by_voltage(self) -> List['FittedMultiData']:
        # for d in self.data:
        v_dict = {d.voltage: d for d in self.data}
        return [FittedMultiData(v_dict[k]) for k in v_dict]

# class ImpedanceFitter():


def fit(data_dir, filename, plot: str = ""):

    array = open_file_numpy(data_dir, filename)
    params = get_params(data_dir, filename)
    print(array.shape)
    data = get_data_numpy(array)
    (freq_log, imped_log, phase, imag_imped, real_imped) = data
    freq = np.power(10, freq_log)

    # np.savetxt("zzz.csv", np.concatenate((freq, real_imped, -imag_imped), axis=1), delimiter=",")
    # f, Z = preprocessing.readCSV("zzz.csv")
    f = freq[:, 0]
    Z = np.array(real_imped + 1j * -imag_imped, dtype=np.complex)[:, 0]

    circuit_str_5a = 'R0-p(R1,CPE1)-p(R2,CPE2)'

    R0_guesses = [6850]
    R1_guesses = [1E7, 1E6, 1E8]
    C1_guesses = [5E-8, 1E-8, 5E-9]
    CPP_guesses = [0.8, 0.9, 1.0]
    R2_guesses = [1E7, 1E5, 1E6, 1E4]
    C2_guesses = [1E-6, 1E-7, 1E-8]
    CPP2_guesses = [0.9, 0.8, 1.0]
    initial_guesses_5a = []

    for R0 in R0_guesses:
        for R1 in R1_guesses:
            for C1 in C1_guesses:
                for CPP in CPP_guesses:
                    for R2 in R2_guesses:
                        for C2 in C2_guesses:
                            for CPP2 in CPP2_guesses:
                                initial_guesses_5a.append([R0, R1, C1, CPP, R2, C2, CPP2])
    num_init_guesses = len(initial_guesses_5a)
    print(f"{num_init_guesses} guess combinations for {circuit_str_5a}")
    assert all(initial_guesses_5a), "Cannot use 0 for any initial guesses"
    chi_sentinal = 0.1
    num_remaining = num_init_guesses
    fitted_dict = {}
    for initial_guess in initial_guesses_5a:

        # print(params.voltage, initial_guess)
        num_remaining -= 1
        circuit = CustomCircuit(circuit=circuit_str_5a, initial_guess=initial_guess)
        circuit.fit(f, Z, global_opt=True, bounds=(
            # R0        R1      C1     CPP1         R2      C2      CPP2      R3    C3      CPP3"
            [6800,      0,       1E-9,      0.8,      0,    0,      0.8],
            [7000,      np.inf,  1E-6,         1, np.inf,    1E-5,    1]
        ))

        fitted_params = circuit.parameters_
        R0, R1, C1, CPP, R2, C2, CPP2 = circuit.parameters_
        Z_fit = circuit.predict(f)

        dof = len(Z)-len(initial_guess)                        # Degrees of freedom - i.e. number of data points - number of variables

        Re_Z = np.real(Z)                                        # Extract real part of Z
        Re_Z_fit = np.real(Z_fit)                                # Extract real part of Z_fit
        variance_re = sum((Re_Z-Re_Z.mean())**2) / (len(Re_Z)-1)    # Variance
        Chi_square_re = sum(((Re_Z-Re_Z_fit)**2)/variance_re)

        Im_Z = np.imag(Z)                                        # Extract real part of Z
        Im_Z_fit = np.imag(Z_fit)                                # Extract real part of Z_fit
        variance_im = sum((Im_Z-Im_Z.mean())**2) / (len(Im_Z)-1)    # Variance
        Chi_square_im = sum(((Im_Z-Im_Z_fit)**2)/variance_im)

        Zsq = abs(Z)
        Zsq_fit = abs(Z_fit)
        variance = sum((Zsq-Zsq.mean())**2) / (len(Z)-1)    # Variance
        Chi_square = sum(((Zsq-Zsq_fit)**2) / variance)

        Chi_sq = Chi_square
        reduced_Chi_sq = Chi_square/dof     # Reduced Chi squared
        std_err = np.sqrt(Chi_sq) * 100
        fitted_dict[tuple(initial_guess)] = (fitted_params, Chi_sq,
                                             Chi_square_re, f, Z, Z_fit, circuit_str_5a, params.voltage)

        divisor = 60 if num_init_guesses > 60 else num_init_guesses
        if num_remaining in range(1, num_init_guesses, int(num_init_guesses / divisor)):
            print(f"{initial_guess}  -  {num_remaining} guesses left")

        # if chi_square_diag < chi_sentinal or chi_square_diag < 0.005 or Chi_square_re < chi_re_sentinal or Chi_square_re < 0.0038:
        if Chi_sq < chi_sentinal:
            chi_sentinal = Chi_sq * 1
            print(f"Fit {params.voltage}, red_chi2={Chi_sq:.8f}, chi2_re={Chi_square_re:.3f}, chi2_im={Chi_square_im:.3f}, std_err={std_err:.4f}, circuit={circuit_str_5a}\t {num_remaining} guesses left")

    fitted_tuple = sorted(fitted_dict.items(), key=lambda item: item[1][1])
    sorted_fitted_dict = OrderedDict(fitted_tuple)
    # pprint.pprint(sorted_fitted_dict)
    return sorted_fitted_dict


def multi_fit(data_dir: str, plot: str = "both", output_dir: str = "") -> FittedMultiData:
    """Fits all files in a folder.
    if plot = "best", plots best fitted data
    if plot = "all", plots all
    if plot = "", plot none
    """
    data_path = Path(data_dir)

    output_path = Path(output_dir) if output_dir else data_path.parent / "output"
    group_fitted = {}
    group_fitted_list = []

    for filepath in list(data_path.glob('*.csv'))[0:]:
        filename = str(filepath.name)
        array = open_file_numpy(data_dir, filename)
        data = get_data_numpy(array)
        (freq_log, imped_log, phase, imag_imped, real_imped) = data
        single_data = SingleData(data=data, params=get_params(data_dir, filename))
        # print(single_data)
        fitted_dict = fit(data_dir, filename, plot=plot)

        if plot in ("all", "both"):
            number_printed = 10
        elif plot == "best":
            number_printed = 1
        else:
            number_printed = 1

        sortedbychi2 = list(fitted_dict.values())[0:number_printed]
        sortedbychi2_re = sorted(fitted_dict.values(), key=lambda item: item[2])[0:number_printed]

        for i, x in enumerate(sortedbychi2 + sortedbychi2_re):
            loop_fit, chi2_loop, chi2_re_loop, f_array, Z_loop, Z_fit_loop, circuit_str, voltage = x
            fit_dc = FittedSingleData(stored_data=single_data, fit_params=loop_fit,
                                      chi2=chi2_loop, circuit=circuit_str, Z_fit=Z_fit_loop)
            R0, R1, C1, CPP, R2, C2, CPP2 = loop_fit

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plot_nyquist(ax, Z_loop, fmt='o')
            plot_nyquist(ax, Z_fit_loop, fmt='-')
            plt.legend(['Data', 'Fit'])
            plt.title(f"{voltage}_{chi2_loop:.8f}")
            fig.set_size_inches(8.5, 5.5, forward=True)

            plt.gcf().text(
                1.1, 0.1,
                f"R0={R0: .0f}\n\n R1={R1: .0f}\n C1={C1}\n CPE1={CPP}\n\n  R2={R2: .0f}\n C2={C2}\n CPE2={CPP2}\n\n  {circuit_str}\n\n chi2v={chi2_loop}",
                fontsize=12)

            chi_str = f"{chi2_loop}".replace('.', '_')
            plt.savefig(Rf'{output_path}\\2R_ss\{voltage}_chi_{chi_str}_{circuit_str}.png',
                        bbox_inches='tight', dpi=500, transparent=True)

            if i in (0, number_printed) and plot in ("best", "both"):
                plt.savefig(Rf'{output_path}\best\{voltage}_chi_{chi_str}_{circuit_str}_b.png',
                            bbox_inches='tight', dpi=500, transparent=True)
                with open(R".\res_best.csv", 'a') as data_f:
                    data_f.write(f"""{fit_dc.voltage}, \t  \n {chi2_loop} {R0}, {R1}, {C1}, {CPP}, {R2}, {C2},
                             {CPP2} \n""")
            plt.close()
            # plt.show()
            group_fitted[fit_dc.voltage] = {
                'circuit_str': circuit_str,
                'fit': loop_fit,
                'chi2': chi2_loop,
                'data': (f_array, Z_loop, Z_fit_loop),
                'voltage': voltage,
                'invsqc1': 1 / (C1 ** 2),
                'invsqc2': 1 / (C2 ** 2)
            }
            # group_fitted_json = json.dumps(group_fitted, sort_keys=True, indent=4, separators=(',', ': '))
            # with open(Rf'{data_dir}\fitted_data.json', 'w') as f:
            # f.write(group_fitted_json)
            group_fitted_list.append(fit_dc)
    group_fitted_dc = FittedMultiData(group_fitted_list)
    return group_fitted_dc


def mott_plot(data_dir, plot="best"):
    data_dc = multi_fit(data_dir, plot=plot)
    voltages = data_dc.voltages
    invsqc1s = data_dc.invsqc1s
    invsqc2s = data_dc.invsqc2s
    """
    voltages = list(data_dict.keys())
    invsqc1s = [x['invsqc1'] for x in data_dict.items()]
    invsqc2s = [x['invsqc2'] for x in data_dict.items()]
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(voltages, invsqc1s)
    # plt.title(f"{params.voltage}, chi2={chi2}")
    plt.savefig(Rf'{data_dir}\mottschottCrandles.png',
                bbox_inches='tight', dpi=500, transparent=True)

    fig2, ax2 = plt.subplots()
    ax2.plot(voltages, invsqc2s)
    # plt.title(f"{params.voltage}, chi2={chi2}")
    plt.savefig(Rf'{data_dir}\mottschottC2randles.png',
                bbox_inches='tight', dpi=500, transparent=True)
    plt.show()


data_dir = str(Path.cwd() / 'input')
multi_fit(data_dir, plot="both")
# mott_plot(data_dir, plot="best")
