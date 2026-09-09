from eng_id_nmt_basic.translate import translate, translate_beam_search
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt
import yaml

# Prepare translation model
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = config["paths"]["best_model_path"]

# English to Indonesian translate funstion
def translatefn():
    sentence = text_box.toPlainText().strip()
    if sentence:
        translation = translate(model, sentence, max_length=100)
        output_label.setText(translation)
        print(translation)

# Create the application
app = QApplication(sys.argv)

# Create a window
window = QWidget()
window.setWindowTitle("English to Indonesian Translation")
window.resize(800, 400)

# Create a button
button = QPushButton("Translate", window)
button.move(670, 350)

# Create text box for input text
text_box = QTextEdit()

# Label for output text
output_label = QLabel("")
output_label.setStyleSheet("""
    QLabel {
        background-color: white;
        border: 1px solid gray;
        border-radius: 5px;
        padding: 10px;
    }
""")
output_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

# Add layouts
main_layout = QVBoxLayout()
# Top layout
top_layout = QHBoxLayout()
# Top layout - Input box
input_layout = QVBoxLayout()
input_title = QLabel("English")
input_layout.addWidget(input_title, 1)
input_layout.addWidget(text_box, 5)
input_frame = QFrame()
input_frame.setLayout(input_layout)
input_frame.setFrameShape(QFrame.Box)
# Top layout - Output box
output_layout = QVBoxLayout()
output_title = QLabel("Indonesian")
output_layout.addWidget(output_title, 1)
output_layout.addWidget(output_label, 5)
output_frame = QFrame()
output_frame.setLayout(output_layout)
output_frame.setFrameShape(QFrame.Box)
top_layout.addWidget(input_frame, 1)
top_layout.addWidget(output_frame, 1)
# Bottom layout
bottom_layout = QHBoxLayout()
bottom_layout.addWidget(button)
# Main layout
main_layout.addLayout(top_layout)
main_layout.addLayout(bottom_layout)
window.setLayout(main_layout)

# Button is clicked -> Translate words
button.clicked.connect(translatefn)

window.show()

sys.exit(app.exec_())