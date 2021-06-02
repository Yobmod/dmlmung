from pathlib import Path
from matplotlib import pyplot as plt
from plot import (multi_bode_axes, multi_bode_add_data, multi_bode_plot_save,
                  get_json_settings,
                  multi_nyquist_add_data, multi_nyquist_axes, multi_nyquist_plot_save)

from mung import open_file_numpy, get_params, get_data_numpy, write_imp_data, write_zview_data
import snoop

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
        print(params.voltage)
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

    print(voltages)

    # plt.show()


main()
