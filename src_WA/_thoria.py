from pathlib import Path
from WA import plot_bet_multi, table_bet_multi, water_added_to_pp, freq_change_to_WA, PlotInputTDict, WA_dataset, plot_multi

x_tho2_ox = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_tho2_ox = freq_change_to_WA(
    [2 * x for x in [17.0, 11.2,
                     20.15, 23.1,
                     26.5, 32.5,
                     29.5, 30.2, 34.2, 33.4, 38.3, 41.1, 63.1, 52.5, 82.9, 156.0]],
    mass_mat=70.5, F0=5794505
)


x_tho2_n = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_tho2_n = freq_change_to_WA(
    [],
    mass_mat=0, F0=0
)

thoria_data_500: PlotInputTDict = {
    "title": "Urania from oxalate vs nitrate @ 500°C",
    "data": [
        WA_dataset(x_tho2_ox, y_tho2_ox, "ThO\u2082 (oxalate)", color="g", mass=70.5, temp=75),
        # WA_dataset(x_tho2_n, y_tho2_n, "ThO\u2082 (nitrate)", color="b", mass=46, temp=75),
    ],
}

sp = Path(R".\results")
thoria_plot_500 = plot_multi(input=thoria_data_500,
                             save_path=sp,
                             file_name="plot_thoria.png",
                             fix_n=False, vmax_line=False, equation=False)
thoria_plot_500_fix = plot_multi(input=thoria_data_500,
                                 save_path=sp,
                                 file_name="plot_thoria_fix.png",
                                 fix_n=True, vmax_line=False, equation=False)
thoria_bet_500 = plot_bet_multi(input=thoria_data_500, save_path=sp,
                                file_name="bet_thoria.png", y_max=1e13)
table_bet_multi(thoria_data_500, save_path=sp, file_name="bet_thoria.tsv")
