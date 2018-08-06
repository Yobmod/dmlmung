# from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter.simpledialog import Dialog
from tkinter import X, TOP, BOTTOM, LEFT, RIGHT, HORIZONTAL, VERTICAL, END, BOTH, ACTIVE, WORD, N, E, NSEW, EW
from PIL import Image, ImageTk
import inspect

# Icons sourced from:
#    http://findicons.com/icon/69404/deletered?width=16#
#    http://findicons.com/icon/93110/old_edit_find?width=16#
class MsgPanel(ttk.Frame):
    def __init__(self, master: tk.Tk, msgtxt: str) -> None:
        ttk.Frame.__init__(self, master)
        self.master = master
        self.pack(side=TOP, fill=X)

        msg = tk.Label(self, wraplength='4i', justify=LEFT)
        msg['text'] = ''.join(msgtxt)
        msg.pack(fill=X, padx=5, pady=5)


class SeeDismissPanel(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        ttk.Frame.__init__(self, master)
        self.master = master
        self.pack(side=BOTTOM, fill=X)       # resize with parent
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        sep = ttk.Separator(orient=HORIZONTAL)       # separator widget
        sep.grid(in_=self, row=0, columnspan=4, sticky=EW, pady=5)

        im_delete = Image.open('images/deletered.png')   # image file
        self.imd = ImageTk.PhotoImage(im_delete)            # handle to file
        dismissBtn = ttk.Button(text='Dismiss', image=self.imd, command=self.winfo_toplevel().destroy)         # Dismiss button
        dismissBtn['compound'] = LEFT  # display image to left of label text
        dismissBtn.grid(in_=self, row=1, column=1, sticky=E)

        im_seecode = Image.open('images/old_edit_find.png')
        self.imsc = ImageTk.PhotoImage(im_seecode)
        codeBtn = ttk.Button(text='See Code', image=self.imsc, default=ACTIVE, command=lambda: CodeDialog(self.master))         # 'See Code' button
        codeBtn['compound'] = LEFT
        codeBtn.grid(in_=self, row=1, column=0, sticky=E)
        # codeBtn.focus()

        self.winfo_toplevel().bind('<Return>', lambda x: codeBtn.invoke())  # < Return > activates 'See Code' button
        self.winfo_toplevel().bind('<Escape>', lambda x: dismissBtn.invoke())  # <'Escape'> activates 'Dismiss' button


class CodeDialog(Dialog):
    """Create a modal dialog to display a demo's source code file. """

    def body(self, master: tk.Tk) -> None:
        """Overrides Dialog.body() to populate the dialog window with a scrolled text window
        and custom dialog buttons. """
        fileName = inspect.getsourcefile(self.parent._create_widgets)  # get t path of  object's parent source code file
        self.master = master
        self.title('Source Code: ' + fileName)

        # create scrolled text widget
        txtFrame = ttk.Frame(self)
        txtFrame.pack(side=TOP, fill=BOTH)
        text = tk.Text(txtFrame, height=24, width=100, wrap=WORD,
                    setgrid=1, highlightthickness=0, pady=2, padx=3)
        # xscroll = ttk.Scrollbar(txtFrame, command=text.xview, orient=HORIZONTAL)
        yscroll = ttk.Scrollbar(txtFrame, command=text.yview, orient=VERTICAL)
        text.configure(
            # xscrollcommand=xscroll.set, 
            yscrollcommand=yscroll.set
            )

        # position in frame and set resize constraints
        text.grid(row=0, column=0, sticky=NSEW)
        yscroll.grid(row=0, column=1, sticky=NSEW)
        txtFrame.rowconfigure(0, weight=1)
        txtFrame.columnconfigure(0, weight=1)

        # add text of file to scrolled text widget
        text.delete('0.0', END)
        text.insert(END, open(fileName).read())


    def buttonbox(self) -> None:
        """Overrides Dialog.buttonbox() to create custom buttons for this dialog. """
        box = ttk.Frame(self)
        box.pack()
        
        cancelBtn = ttk.Button(box, text='Cancel', command=self.cancel)
        cancelBtn.pack(side=RIGHT, padx=5, pady=5)
        self.bind('<Return>', self.cancel)
        self.bind('<Escape>', self.cancel)

