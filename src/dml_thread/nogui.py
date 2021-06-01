import os
from matplotlib import pyplot as plt
from dml_thread.plot import make_multi_bode_plot, multi_bode_title, multi_bode_axes, multi_bode_add_data, get_json_settings
from dml_thread.mung import open_file_numpy, get_params, get_data_numpy, write_imp_data, write_zview_data

data_dir = os.getcwd()

settings = get_json_settings('settings')

plot_exists = False
for file in data_dir:
    params = get_params(file)  # get parameters from file name

    if not plot_exists:
        bode = multi_bode_axes(settings)  # create axes
        # multi_bode_title(bode, params, settings)

    df = open_file_numpy(data_dir, file)  # open file if csv and read into numpy array
    data = get_data_numpy(df)  # return impedance (5 columns) or cv data (4 columns)
    write_imp_data(data)  # writes to csv
    write_zview_data(data)
    multi_bode_add_data(bode, )
    plt.show()
