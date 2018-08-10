import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
import sys
from typing import Union

class Application(ttk.Frame):
    def __init__(self, master: Union[ttk.Frame, tk.Frame, tk.Tk]=None) -> None:
        tk.Frame.__init__(self,master)
        self.createWidgets()

    def createWidgets(self) -> None:
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        canvas=FigureCanvasTkAgg(fig,master=root)
        canvas.get_tk_widget().grid(row=0,column=1)
        canvas.show()

        self.plotbutton=tk.Button(master=root, text="plot", command=lambda: self.polar_plot(canvas,ax))
        self.plotbutton.grid(row=1,column=1)


    def polar_plot(self, canvas: FigureCanvasTkAgg, ax: Figure) -> None:
        c = ['r','b','g']  # plot marker colors
        ax.clear()         # clear axes from previous plot
        for i in range(3):  # plots 3 random data sets
            theta = np.random.uniform(0,360,10)
            r = np.random.uniform(0,1,10)
            ax.plot(theta,r,linestyle="None",marker='o', color=c[i])
            canvas.draw()

root=tk.Tk()
app=Application(master=root)
app.mainloop()
