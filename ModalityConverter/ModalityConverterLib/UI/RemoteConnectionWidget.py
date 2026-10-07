"""Compact remote connection controls used in the Advanced section."""

import os
from qt import QWidget

class RemoteConnectionWidget(QWidget):
    def __init__(self, parent=None):
        super(RemoteConnectionWidget, self).__init__(parent)
        from qt import (
            QLabel, QVBoxLayout, QHBoxLayout,
            QComboBox, QPushButton, QIcon, QSize, QGroupBox
        )

        iconDir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "Resources", "Icons"
            )
        )

        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        # Title OUTSIDE the box
        self.currentResourcesLabel = QLabel("Remote connection")
        self.currentResourcesLabel.setStyleSheet(
            "font-weight: bold; margin-top: 4px;"
        )
        mainLayout.addWidget(self.currentResourcesLabel)

        # Box containing the controls
        self.connectionBox = QGroupBox()
        layout = QVBoxLayout(self.connectionBox)

        # Site selector
        siteRow = QHBoxLayout()
        siteRow.addWidget(QLabel("Site"))

        self.siteSelector = QComboBox()
        self.siteSelector.addItem("Local", None)
        siteRow.addWidget(self.siteSelector, 1)

        self.manageButton = QPushButton()
        self.manageButton.setToolTip("Manage remote sites")
        self.manageButton.setAccessibleName("Manage remote sites")
        self.manageButton.setIcon(
            QIcon(os.path.join(iconDir, "settings.png"))
        )
        self.manageButton.setIconSize(QSize(16, 16))
        self.manageButton.setFixedSize(30, 28)
        self.manageButton.setStyleSheet(
            "background-color: rgb(153, 203, 252);"
        )
        siteRow.addWidget(self.manageButton)

        layout.addLayout(siteRow)

        # Connect button
        self.connectButton = QPushButton("Connect  ")
        self.connectButton.setIcon(
            QIcon(os.path.join(iconDir, "test-connection.png"))
        )
        self.connectButton.setIconSize(QSize(16, 16))
        self.setConnected(False)
        layout.addWidget(self.connectButton)

        # Status
        self.statusLabel = QLabel("● Offline · Local environment only")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setStyleSheet(
            "color: #b36b00; font-weight: 600;"
        )
        layout.addWidget(self.statusLabel)

        mainLayout.addWidget(self.connectionBox)

    def setConnected(self, connected):
        if connected:
            self.connectButton.setText("Disconnect  ")
            self.connectButton.setStyleSheet(
                "background-color: rgba(210, 55, 55, 185);"
            )
        else:
            self.connectButton.setText("Connect  ")
            self.connectButton.setStyleSheet(
                "background-color: rgb(153, 203, 252);"
            )
