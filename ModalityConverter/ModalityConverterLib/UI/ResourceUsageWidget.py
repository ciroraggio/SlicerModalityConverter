"""Resource usage indicators used in ModalityConverter's Advanced section."""

from qt import QLabel, QProgressBar, QWidget, QVBoxLayout


class ResourceUsageWidget(QWidget):
    """Build and update the CPU, RAM, and GPU resource indicators."""

    def __init__(self, parent, formLayout):
        super(ResourceUsageWidget, self).__init__(parent)
        self.bars = {}
        self.labels = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Current resources")
        title.setStyleSheet("font-weight: 600; margin-top: 4px;")
        layout.addWidget(title)

        for key, name in (("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU")):
            label = QLabel("{} used: N/D".format(name))
            label.setStyleSheet("font-size: 10px;")
            bar = QProgressBar(self)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(9)
            layout.addWidget(label)
            layout.addWidget(bar)
            self.labels[key] = label
            self.bars[key] = bar

        formLayout.addRow(self)

    def setResource(self, key, value, labelText=None):
        bar = self.bars[key]
        available = value is not None
        bar.setEnabled(available)
        bar.setValue(max(0, min(100, int(value or 0))))
        if labelText is None:
            labelText = "{} used: {:.0f}%".format(key.upper(), value) if available else "{} used: N/D".format(key.upper())
        self.labels[key].setText(labelText)
