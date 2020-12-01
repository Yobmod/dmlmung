from pathlib import Path
from WA import plot_bet_multi, table_bet_multi, water_added_to_pp, freq_change_to_WA, PlotInputTDict, WA_dataset, plot_multi

# CeO2 10
x_ceo2_ox = water_added_to_pp(
    [10, 10, 10, 10,
    25, 25, 25, 25,
    50, 50, 50, 50,
    75, 75, 75, 75, 
    100, 100, 100, 100, 
    150, 150, 150, 150,
    200, 200, 200, 200, 
    240, 240,240, 240
    ])
y_ceo2_ox = freq_change_to_WA(
    [14.6, 11.1,  16.6, 9.1,
    19.8, 16.3,  21.8, 19.0,
    23.4, 21.5, 24.4, 22.5,
    28.8, 27.9, 30.8, 28.9,
    29.5, 36.6, 31.5, 30.6,
    43.2, 39.5, 32.2, 30.5,
    57.4, 61.3, 41.4, 44.3,
    90.2, 154.8, 52.2, 56.8],
    mass_mat=69, F0=5791070
    )


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
    [0.50* x for x in [
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

ceria_data_500: PlotInputTDict = {
    "title": "Ceria oxalate vs nitrate",
    "data": [
        WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO\u2082 (oxalate)", color="c", mass=69, temp=75),
        WA_dataset(x_ceo2_n, y_ceo2_n, "CeO\u2082 (nitrate)", color="b", mass=46, temp=75),
    ],
}

sp = Path(R".\results")
ceria_plot_500 = plot_multi(input=ceria_data_500,
                                save_path=sp,
                                file_name="plot_ceria.png",
                                fix_n=False, vmax_line=False, equation=False)
ceria_plot_500_fix = plot_multi(input=ceria_data_500,
                                save_path=sp,
                                file_name="plot_ceria_fix.png",
                                fix_n=True, vmax_line=False, equation=False)
oxalate_bet_500 = plot_bet_multi(input=ceria_data_500, 
                                save_path=sp, file_name="bet_ceria.png", y_max=1e13)
table_bet_multi(ceria_data_500, save_path=sp,file_name="bet_ceria.tsv")