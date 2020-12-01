from WA import water_added_to_pp, freq_change_to_WA, PlotInputTDict, WA_dataset, plot_multi

# CeO2 10
x_ceo2_ox = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_ceo2_ox = freq_change_to_WA(
    [146, 91, 198, 170, 224, 205, 288, 269, 295, 286, 312, 295, 374, 403, 502, 548],
    mass_mat=69, F0=5791070
)

x_ceo2_n = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_ceo2_n = freq_change_to_WA(
    [52, 39, 82, 111, 112, 129, 123, 140, 134, 161, 158, 160, 174, 227, 400, 312],
    mass_mat=46, F0=5751200
)

ceria_data_500: PlotInputTDict = {
    "title": "Ceria oxalate vs nitrate",
    "data": [
        WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO\u2082 (oxalate)", color="c"),
        WA_dataset(x_ceo2_n, y_ceo2_n, "CeO\u2082 (nitrate)", color="b"),
    ],
}

ceria_plot_500 = plot_multi(input=ceria_data_500, file_name="plot_ceria.png", fix_n=True, vmax_line=True)
