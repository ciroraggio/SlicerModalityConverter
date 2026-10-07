"""Dialog for creating and editing multiple remote server profiles."""

import os

import qt


class RemoteSitesDialog(qt.QDialog):
    def __init__(self, sites=None, parent=None):
        super(RemoteSitesDialog, self).__init__(parent)
        self.setWindowTitle("Remote sites")
        self.sites = []
        self._rows = []
        self._iconDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Resources", "Icons"))

        layout = qt.QVBoxLayout(self)
        layout.addWidget(qt.QLabel("Add one or more remote sites. Select a site and connect from the \"Remote connections\" widget."))
        self._addButton = qt.QPushButton("+ Add new site")
        self._addButton.setStyleSheet("background-color: rgb(153, 203, 252);")
        layout.addWidget(self._addButton)
        self._addButton.clicked.connect(lambda checked=False: self._addRow())

        buttons = qt.QHBoxLayout()
        buttons.addStretch(1)
        self._cancelButton = qt.QPushButton("Cancel")
        self._cancelButton.setStyleSheet("background-color: rgba(210, 55, 55, 185);")
        self._saveButton = qt.QPushButton("Save")
        self._saveButton.setStyleSheet("background-color: rgb(52, 206, 165);")
        buttons.addWidget(self._cancelButton)
        buttons.addWidget(self._saveButton)
        layout.addLayout(buttons)
        self._saveButton.clicked.connect(lambda checked=False: self._save())
        self._cancelButton.clicked.connect(lambda checked=False: self.reject())

        for site in sites or []:
            self._addRow(site)
        self.adjustSize()

    def _addRow(self, site=None):
        row = qt.QWidget(self)
        rowLayout = qt.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        fields = [qt.QLineEdit() for _ in range(4)]
        nameEdit, hostEdit, portEdit, tokenEdit = fields
        nameEdit.setPlaceholderText("Site name")
        nameEdit.setFixedWidth(95)
        hostEdit.setPlaceholderText("Host or URL")
        hostEdit.setFixedWidth(140)
        portEdit.setPlaceholderText("Port")
        portEdit.setFixedWidth(55)
        portEdit.setText("8765")
        tokenEdit.setPlaceholderText("Bearer token")
        tokenEdit.setFixedWidth(150)
        tokenEdit.setEchoMode(qt.QLineEdit.Password)
        if site:
            nameEdit.setText(site.get("name", ""))
            hostEdit.setText(site.get("host", ""))
            portEdit.setText(site.get("port", "8765"))
            tokenEdit.setText(site.get("token", ""))
        for field in fields:
            rowLayout.addWidget(field)

        removeButton = qt.QPushButton()
        removeButton.setToolTip("Remove site")
        removeButton.setAccessibleName("Remove site")
        removeButton.setIcon(qt.QIcon(os.path.join(self._iconDir, "trash.png")))
        removeButton.setIconSize(qt.QSize(16, 16))
        removeButton.setFixedSize(28, 28)
        removeButton.setStyleSheet("background-color: rgba(210, 55, 55, 185);")
        rowLayout.addWidget(removeButton)
        item = {"widget": row, "name": nameEdit, "host": hostEdit, "port": portEdit, "token": tokenEdit}
        self._rows.append(item)
        
        # Defer removal until the clicked signal has finished. Removing the
        # sender's parent row while Qt is dispatching the button signal can
        # tear down the dialog (and, in Slicer, the host app).
        removeButton.clicked.connect(lambda checked=False, item=item: qt.QTimer.singleShot(0, lambda item=item: self._removeRow(item)))
        
        self.layout().insertWidget(self.layout().indexOf(self._addButton), row)
        self.adjustSize()

    def _removeRow(self, item):
        if item in self._rows:
            self._rows.remove(item)
            self.layout().removeWidget(item["widget"])
            item["widget"].deleteLater()
            self.adjustSize()

    def _save(self):
        self.sites = []
        for row in self._rows:
            name, host = row["name"].text.strip(), row["host"].text.strip()
            token = row["token"].text.strip()
            if not name and not host and not token:
                continue
            self.sites.append({
                "name": name or host,
                "host": host,
                "port": row["port"].text.strip() or "8765",
                "token": token,
            })
        self.accept()
