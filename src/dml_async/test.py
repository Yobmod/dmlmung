import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from matplotlib.figure import Figure
import tkinter as tk
import tkinter.ttk as ttk
import sys


class Application(ttk.Frame):
    def __init__(self, master: tk.Tk=None) -> None:
        ttk.Frame.__init__(self, master)
        self.createWidgets()

    def createWidgets(self) -> None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=0, column=1)
        canvas.show()

        self.plotbutton = tk.Button(
            master=root, text="plot", command=lambda: self.plot(canvas, ax))
        self.plotbutton.grid(row=0, column=0)

    def plot(self, canvas: FigureCanvasTkAgg, ax:plt) -> None:
        for line in sys.stdout:  # infinite loop, reads data of a subprocess
            theta = line[1]
            r = line[2]
            ax.plot(theta, r, linestyle="None", maker='o')
            canvas.draw()
            ax.clear()
            #here set axes


root = tk.Tk()
app = Application(master=root)
app.mainloop()
