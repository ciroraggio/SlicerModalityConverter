import qt

class HelpDialog(qt.QDialog):
    def __init__(self, parent=None):
        super(HelpDialog, self).__init__(parent)
        from ImageTranslatorLib.UI.utils import HELP_TEXT, CONTRIBUTORS
        self.setWindowTitle("I2IHub - Help Guide and Acknowledgements")
        self.setMinimumWidth(600)
        self.setMinimumHeight(800)

        mainLayout = qt.QVBoxLayout(self)

        # Create tab widget
        tabWidget = qt.QTabWidget()
        mainLayout.addWidget(tabWidget)

        # Help tab
        helpTab = qt.QWidget()
        helpLayout = qt.QVBoxLayout(helpTab)
        helpTextEdit = qt.QTextBrowser()
        helpTextEdit.setOpenExternalLinks(True)
        helpTextEdit.setReadOnly(True)
        helpTextEdit.setHtml(HELP_TEXT)
        helpLayout.addWidget(helpTextEdit)
        tabWidget.addTab(helpTab, "Help Guide")

        # Acknowledgements tab
        ackTab = qt.QWidget()
        ackLayout = qt.QVBoxLayout(ackTab)
        ackTextEdit = qt.QTextEdit()
        ackTextEdit.setReadOnly(True)

        # Format contributors as HTML
        contributors_html = "<h3>Contributors</h3><ul>"
        for contributor in CONTRIBUTORS:
            contributors_html += f"<li>{contributor}</li>"
        contributors_html += "</ul>"
        ackTextEdit.setHtml(contributors_html)
        ackLayout.addWidget(ackTextEdit)
        tabWidget.addTab(ackTab, "Acknowledgements")


