import sys
from PySide2.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
#label = QLabel("Hello World!")
label = QLabel("<font color=red size=40>Hello World!</font>")
label.show()
app.exec_()


import sys
from PySide2.QtWidgets import QApplication, QMessageBox

# Create the application object
app = QApplication(sys.argv)

# Create a simple dialog box
msg_box = QMessageBox()
msg_box.setText("Hello World!")
msg_box.show()

sys.exit(msg_box.exec_())
