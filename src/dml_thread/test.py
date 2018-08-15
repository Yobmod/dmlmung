import tkinter as tk
import tkinter.ttk as ttk

from tkinter.colorchooser import askcolor

class testapp(ttk.Frame):

    def __init__(self, master: tk.Tk = None) -> None:
        super().__init__(master)

        if master is not None:
            self.master = master

        self.create_widgets()

    def create_widgets(self):
        self.color()

    def color(self):
        colRGB, colHex = askcolor(
            parent=self, color="red", title="Color Chooser")



if __name__ == "__main__":
    root = tk.Tk()
    app = testapp(root)

    app.mainloop()
