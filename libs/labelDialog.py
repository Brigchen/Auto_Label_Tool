try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.utils import newIcon, labelValidator

BB = QDialogButtonBox


class LabelDialog(QDialog):

    def __init__(self, text="Enter object label", parent=None, listItem=None):
        super(LabelDialog, self).__init__(parent)

        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(labelValidator())
        self.edit.editingFinished.connect(self.postProcess)

        model = QStringListModel()
        model.setStringList(listItem)
        completer = QCompleter()
        completer.setModel(model)
        self.edit.setCompleter(completer)

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        self.listWidget = None
        if listItem is not None and len(listItem) > 0:
            self.listWidget = QListWidget(self)
            for item in listItem:
                self.listWidget.addItem(item)
            self.listWidget.itemClicked.connect(self.listItemClick)
            self.listWidget.itemDoubleClicked.connect(self.listItemDoubleClick)
            self.listWidget.itemActivated.connect(self._on_list_activated)
            self.listWidget.installEventFilter(self)
            layout.addWidget(self.listWidget)
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)
        self._list_click_accepts = False

    def eventFilter(self, obj, event):
        if self.listWidget is not None and obj is self.listWidget and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._confirm_list_selection()
                return True
        return super(LabelDialog, self).eventFilter(obj, event)

    def _confirm_list_selection(self):
        if self.listWidget is None:
            return
        item = self.listWidget.currentItem()
        if item is None and self.listWidget.count() > 0:
            item = self.listWidget.item(0)
        if item is None:
            return
        self.listItemClick(item)
        self.validate()

    def _on_list_activated(self, item):
        if item is not None:
            self.listItemClick(item)
            self.validate()

    def validate(self):
        try:
            if self.edit.text().trimmed():
                self.accept()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            if self.edit.text().strip():
                self.accept()

    def postProcess(self):
        try:
            self.edit.setText(self.edit.text().trimmed())
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            self.edit.setText(self.edit.text())

    def _select_list_by_text(self, text):
        if self.listWidget is None:
            return
        key = (text or '').strip()
        for i in range(self.listWidget.count()):
            it = self.listWidget.item(i)
            if it.text().strip() == key:
                self.listWidget.setCurrentItem(it)
                self.listWidget.scrollToItem(it)
                return

    def popUp(self, text='', move=True, focus_list=None, list_click_accepts=False,
              list_highlight=None):
        self._list_click_accepts = bool(list_click_accepts)
        self.edit.setText(text)
        use_list = (
            focus_list
            if focus_list is not None
            else self.listWidget is not None and self.listWidget.count() > 0
        )
        if use_list:
            highlight = list_highlight if list_highlight is not None else text
            self._select_list_by_text(highlight)
            self.listWidget.setFocus(Qt.PopupFocusReason)
        else:
            self.edit.setSelection(0, len(text))
            self.edit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self.edit.text() if self.exec_() else None

    def listItemClick(self, tQListWidgetItem):
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self.edit.setText(text)
        if self._list_click_accepts:
            self.validate()

    def listItemDoubleClick(self, tQListWidgetItem):
        self.listItemClick(tQListWidgetItem)
        self.validate()
