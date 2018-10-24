
### 0.0--------------------------------------

Create project:
            Git:
                Readme
                Gitignore
                Gitatributes
                Licence

                
            Pipfile:
                Pylama
                Mypy
                Coverage
                Pytest

            Requirements.txt
            Setup.cfg
            Workplan.md



0.1---------------------------------------

Scripts to:
        - open CV files & check formatting

    - Mung data with numpy

    - Create & save CV graphs with Matplotlib



0.2----------------------------------------

Scripts to:
        - open Imp files & check formatting

    - Mung data

    - Create & save Bose plots

    - Create & save Nyquist  plots

    - Save imp data as CSV


0.3----------------------------------------

Tkinter GUI:
    Buttons to run scripts


Add cython test


0.4---------------------------------------

Tkinter GUI:
    Buttons to select folders / files
    Import settings from JSON

Try Asyncio 


0.5-----------------------------------------

Pyinstaller:
    - Create spec file
    - Add zip script with shututil

Try Thread/Processpool
Nb: single process/thread = 6.40/2
    multiprocess = 4.80/2
    cython single process no types: 6.35/2

0.6-------------------------------------------


Add Processpool

Make installer with NSIS


### 0.7--------------------------------------------

Tkinter GUI:

    Add notebook pages

            Settings

                Add button to view/edit settings file

                Add revert to default settings button

            Batch

                Select folder

                Go

            File

                Select file

                Go

            Misc

                (for testing)

### 0.8---------------------------------------------

Tkinter GUI:

        Add canvas to file page

        Display graph on canvas

    Add text fields to update settings

    Add color picker to update settings

    Display print/log(msges) in a text box


Pyinstaller make unwindowed


### 0.9------------------------------------------------

Tkinter GUI:

        Select boxes for make of potentiometer

Scripts: 

        Add potentiometer logic



1.0-----------------------------------------------

Logging

Pytest tests

Release!

1.1------------------------------------------------

Make File single thread

    Checkbox for parallel if more than one file selected


Make Batch recursive with checkbox

    Add timeout / recursion limit

1.1+ ----------------------------------------------

Cythonize:

    numpy arrays?

    File IO?

Gevent/Threads/Trio?:

    File IO

Pyside/Kivy/Pygame?:

    GUI

Combine with dmlQCM

