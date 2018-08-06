import tkinter as tk
from tkinter import X, Y, TOP, BOTTOM, LEFT, RIGHT, HORIZONTAL, VERTICAL, END, BOTH, ACTIVE, WORD, N, E, NSEW, EW
from tkinter import ttk
import panels

#import panels

class dmlNotebook(ttk.Frame):
    """"""
    def __init__(self, master: tk.Tk=None, name: str='dmlMung') -> None:
        super().__init__(master, name=name)
        if master is not None:
            self.master = master
        self.master.title('Dml Mung')
        self._create_widgets()
        self.pack(expand=Y, fill=BOTH)

        style = ttk.Style()
        style.theme_create("MyStyle", parent="alt", settings={
                "TNotebook": {"configure": {"tabmargins": [5, 5, 5, 0]}}, # gaps L, T, R, B
                "TNotebook.Tab": {"configure": {"padding": [10, 5]}, }}) # tab size [x, y]
        style.theme_use("MyStyle")

    def _create_widgets(self) -> None:
        panels.SeeDismissPanel(self)
        self._create_notebook_panel()


    def _create_notebook_panel(self) -> None:
        MainPanel = tk.Frame(self, name='demo')
        MainPanel.pack(side=TOP, fill=BOTH, expand=Y)
 
        # create the notebook
        nb = ttk.Notebook(MainPanel, name='notebook')
        nb.enable_traversal()  # extend bindings to top level window (CTRL+TAB, SHIFT+CTRL+TAB, ALT+K )
        nb.pack(fill=BOTH, expand=Y, padx=2, pady=3)

        # create tabs in notebook
        self._create_home_tab(nb)
        self._create_file_tab(nb)
        self._create_batch_tab(nb)
        self._create_settings_tab(nb)
        # self._create_disabled_tab(nb)
        self._create_text_tab(nb)
        self._create_misc_tab(nb)


    def _create_home_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Home'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='   Home   ', underline=4)


    def _create_file_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'File'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='   File   ', underline=3)


    def _create_batch_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Batch'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='  Batch  ', underline=2)


    def _create_settings_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Settings'"""
        settings_frame = ttk.Frame(nb, name='settings')
        settings_frame.rowconfigure(1, weight=1)
        settings_frame.columnconfigure((0, 1), weight=1, uniform=1)
        nb.add(settings_frame, text='Settings', underline=0, padding=2)  # underline = shortcut char index
        
        msg = ["Please change settings and click save. To return to default settings, click reset. \nSettings will be applied to File and Batch plots"]   # 
        settingslbl = ttk.Label(settings_frame, wraplength='4i', justify=LEFT, anchor=N, text=''.join(msg))
        settingslbl.grid(row=0, column=0, columnspan=2, sticky='new', pady=5)

        loadbtn = ttk.Button(settings_frame, text='Load Settings', underline=0, command=None)
        loadbtn.grid(row=3, column=3, pady=(2, 4))

        savebtn = ttk.Button(settings_frame, text='Save Settings', underline=0, command=None)
        savebtn.grid(row=4, column=3, pady=(2, 4))

        resetbtn = ttk.Button(settings_frame, text='Reset', underline=0, command=None)
        resetbtn.grid(row=5, column=3, pady=(2, 4))


    def _create_misc_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Misc'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='Misc', underline=0)


    def _create_disabled_tab(self, nb: ttk.Notebook) -> None:
        """Populate the second pane. Note that the content doesn't really matter"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='Disabled', state='disabled')


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
    dmlNotebook().mainloop()
