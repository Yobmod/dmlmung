import json
from impedance.visualization import plot_nyquist
from impedance.models.circuits import CustomCircuit
from pathlib import Path
from plot import (multi_bode_axes, multi_bode_add_data, multi_bode_plot_save,
                  get_json_settings,
                  multi_nyquist_add_data, multi_nyquist_axes, multi_nyquist_plot_save)

from mung import open_file_numpy, get_params, get_data_numpy, write_imp_data, write_zview_data
import snoop
import pprint
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Sequence
from dataclasses import dataclass

DEBUG = False
snoop.install(enabled=DEBUG)


@snoop(depth=1)
def main():
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
        pprint(params.voltage)
        df = open_file_numpy(data_dir, filename)  # open file if csv and read into numpy array
        if df.any():
            data = get_data_numpy(df)  # return impedance (5 columns) or cv data (4 columns)
        if data and len(data) == 5:
            write_imp_data(data, params)  # writes to csv
            write_zview_data(data, params)
            (freq_log, imped_log, phase, imag_imped, real_imped) = data

        for version in ["both", "phase", "gain"]:
            if version == "both":
                if not both_plot_exists:
                    bode_both = multi_bode_axes(settings, y_axes=version)  # create axes
                    both_plot_exists = version  # make truthy
                multi_bode_add_data(bode_both, freq_log, imped_log, phase, params)
            elif version == "phase":
                if not phase_plot_exists:
                    bode_phase = multi_bode_axes(settings, y_axes=version)  # create axes
                    phase_plot_exists = version  # make truthy
                    # multi_bode_title(bode_phase, params, settings)
                multi_bode_add_data(bode_phase, freq_log, imped_log, phase, params)

            elif version == "gain":
                if not gain_plot_exists:
                    bode_gain = multi_bode_axes(settings, y_axes=version)  # create axes
                    gain_plot_exists = version  # make truthy
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
    params: Tuple
    data: Tuple

    def __post_init__(self):
        self.freq_log, self.imped_log, self.phase, self.imag_imped, self.real_imped = self.data
        self.imped = self.Z = np.array(self.real_imped + 1j * -self.imag_imped, dtype=np.complex)[:, 0]
        self.freq = np.power(10, self.freq_log)[:, 0]
        self.voltage = self.params.voltage


@dataclass
class fit_single_data():
    stored_data: SingleData
    circuit: str
    fit_params: Tuple
    chi2: float  # calc in DC?

    Z_fit: Sequence
    Re_Z_fit: Sequence  # calc in DC?

    def __post_init__(self):
        self.voltage = self.stored_data.voltage
        self.C1 = self.fit_params[2]
        self.C2 = self.fit_params[5]


@dataclass
class FittedMultiData():
    data: Sequence[fit_single_data]

    def __post_init__(self):
        self.voltages = [d.voltage for d in self.data]
        self.invsqc1s = [1 / (C ** 2 * 1000000000000) for C in self.data.C1]
        self.invsqc2s = [1 / (C ** 2 * 1000000000000) for C in self.data.C2]


def fit(data_dir, filename, plot: str = ""):

    array = open_file_numpy(data_dir, filename)
    params = get_params(data_dir, filename)
    # print(array.shape)
    data = get_data_numpy(array)
    (freq_log, imped_log, phase, imag_imped, real_imped) = data
    freq = np.power(10, freq_log)

    # np.savetxt("zzz.csv", np.concatenate((freq, real_imped, -imag_imped), axis=1), delimiter=",")
    # f, Z = preprocessing.readCSV("zzz.csv")
    f = freq[:, 0]
    Z = np.array(real_imped + 1j * -imag_imped, dtype=np.complex)[:, 0]

    circuit_str_5a = 'R0-p(R1,CPE1)-p(R2,C2)'

    R0_guesses = range(1, 1000, 200)
    R1_guesses = [1E5, 1E6, 1E7, 1E8, 1E9]
    C1_guesses = [1E-6, 1E-7, 1E-8, 1E-9, 1E-10]
    CPP_guesses = [0.8]
    R2_guesses = [1E5, 1E6, 1E7, 1E8, 1E9]
    C2_guesses = [1E-6, 1E-7, 1E-8, 1E-9]
    initial_guesses_5a = []

    """
    initial_guesses_5a = [

        [750, 1.73E6, 6.56E-8, 0.827, 2.088E6, 7.22E-8],
        [750, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-8],
        [700, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-8],


        [750, 1.73E6, 6.56E-7, 0.827, 2.088E6, 7.22E-9],
        [750, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-9],
        [700, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-9],

        [7500, 1.73E6, 6.56E-8, 0.827, 2.088E6, 7.22E-8],
        [7500, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-8],
        [7000, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-8],

        [7500, 1.73E6, 6.56E-7, 0.827, 2.088E6, 7.22E-8],
        [7500, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-8],
        [7000, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-8],

        [7500, 1.73E6, 6.56E-9, 0.827, 2.088E6, 7.22E-8],
        [7500, 1.73E7, 6.56E-9, 0.827, 2.088E6, 7.22E-8],
        [7000, 1.73E7, 6.56E-9, 0.827, 2.088E6, 7.22E-8],

        [7500, 1.73E6, 6.56E-7, 0.827, 2.088E6, 7.22E-9],
        [7500, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-9],
        [7000, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-9],

        [7500, 1.73E6, 6.56E-8, 0.827, 2.088E6, 7.22E-9],
        [7500, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-9],
        [7000, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-9],

        [7500, 1.73E6, 6.56E-9, 0.827, 2.088E6, 7.22E-9],
        [7500, 1.73E7, 6.56E-9, 0.827, 2.088E6, 7.22E-9],
        [7000, 1.73E7, 6.56E-9, 0.827, 2.088E6, 7.22E-9],

        [7500, 1.73E6, 6.56E-7, 0.827, 2.088E6, 7.22E-7],
        [7500, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-7],
        [7000, 1.73E7, 6.56E-7, 0.827, 2.088E6, 7.22E-7],

        [7500, 1.73E6, 6.56E-8, 0.827, 2.088E6, 7.22E-7],
        [7500, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-7],
        [7000, 1.73E7, 6.56E-8, 0.827, 2.088E6, 7.22E-7],

        [7500, 1.73E6, 6.56E-9, 0.827, 2.088E6, 7.22E-7],
        [7500, 1.73E7, 6.56E-9, 0.827, 2.088E6, 7.22E-7],
        [7000, 1.73E7, 6.56E-9, 0.827, 2.088E6, 7.22E-7],
    ]
    """

    for R0 in R0_guesses:
        for R1 in R1_guesses:
            for C1 in C1_guesses:
                for CPP in CPP_guesses:
                    for R2 in R2_guesses:
                        for C2 in C2_guesses:
                            initial_guesses_5a.append([R0, R1, C1, CPP, R2, C2])

    print(f"{len(initial_guesses_5a)} guess combinations")
    assert all(initial_guesses_5a), "Cannot use 0 for any initial guesses"

    fitted_dict = {}
    for initial_guess in initial_guesses_5a:
        print(params.voltage, initial_guess)
        circuit = CustomCircuit(circuit=circuit_str_5a, initial_guess=initial_guess)
        circuit.fit(f, Z, global_opt=False, bounds=(
            [0, 0, 0, 0, 0, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))

        fitted_params = circuit.parameters_
        R0, R1, C1, CPP, R2, C2 = circuit.parameters_
        Z_fit = circuit.predict(f)

        Re_Z = np.real(Z)                                        # Extract real part of Z
        Re_Z_fit = np.real(Z_fit)                                # Extract real part of Z_fit

        dof = len(Re_Z)-len(initial_guess)                        # Degrees of freedom - i.e. number of data points - number of variables
        variance = sum((Re_Z-Re_Z.mean())**2) / (len(Re_Z)-1)    # Variance
        Chi_square = sum(((Re_Z-Re_Z_fit)**2)/variance)          # Chi squared
        red_Chi_sq = 100 * Chi_square / dof  # Reduced Chi squared

        fitted_dict[tuple(initial_guess)] = (fitted_params, red_Chi_sq, f, Z, Z_fit, circuit_str_5a, params.voltage)

        if plot == "all":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plot_nyquist(ax, Z, fmt='o')
            plot_nyquist(ax, Z_fit, fmt='-')
            plt.legend(['Data', 'Fit'])
            plt.title(params.voltage)
            plt.savefig(Rf'data_dir\{params.voltage} chi={red_Chi_sq}).png',
                        bbox_inches='tight', dpi=500, transparent=True)
            # plt.show()

    fitted_tuple = sorted(fitted_dict.items(), key=lambda item: item[1][1])
    sorted_fitted_dict = OrderedDict(fitted_tuple)
    # pprint.pprint(sorted_fitted_dict)
    return sorted_fitted_dict


def multi_fit(data_dir, plot: str = "best"):
    """Fits all files in a folder.
    if plot = "best", plots best fitted data
    if plot = "all", plots all
    if plot = "", plot none
    """

    group_fitted = {}
    group_fitted_list = []

    for filepath in Path(data_dir).glob('*.csv'):
        filename = str(filepath.name)
        single_data = SingleData(data=[], params=get_params(data_dir, filename))

        fitted_dict = fit(data_dir, filename, plot=plot)
        best_fit, chi2, f, Z, Z_fit, circuit_str, voltage = list(fitted_dict.values())[0]

        fit_dc = fit_single_data(stored_data=single_data, fit_params=best_fit,
                                 chi2=chi2, circuit=circuit_str, Z_fit=Z_fit)
        assert fit_dc.voltage == voltage

        if plot == "best":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plot_nyquist(ax, Z, fmt='o')
            plot_nyquist(ax, Z_fit, fmt='-')
            plt.legend(['Data', 'Fit'])
            plt.title(f"{voltage}, chi2={chi2}")
            # plt.show()

            plt.savefig(Rf'{data_dir}\{fit_dc.voltage}_bestfit.png',
                        bbox_inches='tight', dpi=500, transparent=True)
        R0, R1, C1, CPP, R2, C2 = best_fit
        # print(R0, R1, C1, CPP, R2, C2)

        group_fitted[fit_dc.voltage] = {
            'circuit_str': circuit_str,
            'fit': best_fit,
            'chi2': chi2,
            'data': (f, Z, Z_fit),
            'voltage': voltage,
            'invsqc1': 1 / (C1 ** 2 * 1000000000000),
            'invsqc2': 1 / (1 / C2 ** 2 * 1000000000000)
        }
        group_fitted_list.append(fit_dc)

    group_fitted_dc = FittedMultiData(group_fitted_list)
    group_fitted_json = json.dumps(group_fitted, sort_keys=True, indent=4, separators=(',', ': '))
    with open(Rf'{data_dir}\fitted_data.json', 'w') as f:
        f.write(group_fitted_json)

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
    plt.savefig(Rf'{data_dir}\mottschottC1.png',
                bbox_inches='tight', dpi=500, transparent=True)

    fig2, ax2 = plt.subplots()
    ax2.plot(voltages, invsqc2s)
    # plt.title(f"{params.voltage}, chi2={chi2}")
    plt.savefig(Rf'{data_dir}\mottschottC2.png',
                bbox_inches='tight', dpi=500, transparent=True)
    plt.show()


data_dir = str(Path.cwd() / 'input')
mott_plot(data_dir, plot="best")
