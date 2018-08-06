# from tkinter import *
import tkinter as tk
from tkinter import X, Y, TOP, BOTTOM, LEFT, RIGHT, HORIZONTAL, VERTICAL, END, BOTH, ACTIVE, WORD, N, E, NSEW, EW
from tkinter import ttk
from dmltk.panels import MsgPanel, SeeDismissPanel
import tkinter as tk

class NotebookDemo(ttk.Frame):
    """"""
    def __init__(self, isapp: bool=True, master: tk.Tk=None, name: str='notebookdemo') -> None:
        ttk.Frame.__init__(self, name=name)
        self.pack(expand=Y, fill=BOTH)
        if master is not None:
            self.master = master
        self.master.title('Notebook Demo')
        self.isapp = isapp
        self._create_widgets()

    def _create_widgets(self) -> None:
        if self.isapp:
            SeeDismissPanel(self)
        self._create_demo_panel()

    def _create_demo_panel(self) -> None:
        demoPanel = tk.Frame(self, name='demo')
        demoPanel.pack(side=TOP, fill=BOTH, expand=Y)
        nb = ttk.Notebook(demoPanel, name='notebook')     # create the notebook
        # extend bindings to top level window allowing
        #   CTRL+TAB - cycles thru tabs
        #   SHIFT+CTRL+TAB - previous tab
        #   ALT+K - select tab using mnemonic (K = underlined letter)
        nb.enable_traversal()
        nb.pack(fill=BOTH, expand=Y, padx=2, pady=3)
        self._create_descrip_tab(nb)
        self._create_disabled_tab(nb)
        self._create_text_tab(nb)

    def _create_descrip_tab(self, nb: ttk.Notebook) -> None:
        """frame to hold contentx"""
        frame = ttk.Frame(nb, name='descrip')
        msg = ["Ttk is the new Tk themed widget set. One of the widgets ",
               "it includes is the notebook widget, which provides a set ",
               "of tabs that allow the selection of a group of panels, ",
               "each with distinct content. They are a feature of many ",
               "modern user interfaces. Not only can the tabs be selected ",
               "with the mouse, but they can also be switched between ",
               "using Ctrl+Tab when the notebook page heading itself is ",
               "selected. Note that the second tab is disabled, and cannot "
               "be selected."]   # widgets to be displayed on 'Description' tab
        lbl = ttk.Label(frame, wraplength='4i', justify=LEFT, anchor=N, text=''.join(msg))
        neatVar = tk.StringVar()
        btn = ttk.Button(frame, text='Neat!', underline=0, command=lambda v=neatVar: self._say_neat(v))
        neat = ttk.Label(frame, textvariable=neatVar, name='neat')

        # position and set resize behaviour
        lbl.grid(row=0, column=0, columnspan=2, sticky='new', pady=5)
        btn.grid(row=1, column=0, pady=(2, 4))
        neat.grid(row=1, column=1,  pady=(2, 4))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure((0, 1), weight=1, uniform=1)

        # bind for button short-cut key
        # (must be bound to toplevel window)
        self.winfo_toplevel().bind('<Alt-n>', lambda e, v=neatVar: self._say_neat(v))

        # add to notebook (underline = index for short-cut character)
        nb.add(frame, text='Description', underline=0, padding=2)

    def _say_neat(self, v: tk.StringVar) -> None:
        v.set('Yeah, I know...')
        self.update()
        self.after(500, v.set(''))

    # =============================================================================
    def _create_disabled_tab(self, nb: ttk.Notebook) -> None:
        """Populate the second pane. Note that the content doesn't really matter"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='Disabled', state='disabled')

    # =============================================================================
    def _create_text_tab(self, nb: ttk.Notebook) -> None:
        """populate the third frame with a text widget"""
        frame = ttk.Frame(nb)
        txt = tk.Text(frame, wrap=WORD, width=40, height=10)
        vscroll = ttk.Scrollbar(frame, orient=VERTICAL, command=txt.yview)
        txt['yscroll'] = vscroll.set
        vscroll.pack(side=RIGHT, fill=Y)
        txt.pack(fill=BOTH, expand=Y)
        nb.add(frame, text='Text Editor', underline=0)  # add to notebook (underline = index for short-cut character)


if __name__ == '__main__':
    NotebookDemo().mainloop()
