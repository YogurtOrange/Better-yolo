"""Live, native-widget scaling without changing image or annotation coordinates."""

import re
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QAbstractButton, QAbstractItemView, QDialog, QFormLayout,
    QGridLayout, QListView, QMainWindow, QSplitter, QTableView, QWidget,
)


@dataclass
class _ScaleBaseline:
    # A custom Python object stays a PyQt_PyObject in a QObject property. A bare
    # dict would be converted to a QVariantMap and copied when retrieved.
    values: dict


def scale_stylesheet(source, factor):
    """Scale pixel lengths, including inline styles, from an unchanged source."""
    return re.sub(
        r"(-?\d+(?:\.\d+)?)px\b",
        lambda match: f"{round(float(match.group(1)) * factor)}px",
        source,
    )


def ui_pixels(value):
    app = QApplication.instance()
    return round(value * (app.property("uiScaleFactor") or 1.0)) if app else value


def ui_font(family, points, weight=QFont.Weight.Normal):
    font = QFont(family, points, weight)
    app = QApplication.instance()
    font.setPointSizeF(points * (app.property("uiScaleFactor") or 1.0) if app else points)
    return font


class UiScaleManager(QObject):
    """Keep logical metrics so repeated zoom changes never accumulate rounding."""

    scale_changed = pyqtSignal(int)
    MIN_PERCENT = 75
    MAX_PERCENT = 200

    def __init__(self, app, stylesheet, parent=None):
        super().__init__(parent)
        self.app = app
        self.stylesheet = stylesheet
        self.percent = 100
        self._widgets = WeakKeyDictionary()
        self._layouts = WeakKeyDictionary()
        self._applying = False
        self._pending = False
        app.setProperty("uiScaleFactor", 1.0)
        app.setStyleSheet(stylesheet)
        app.installEventFilter(self)

    @property
    def factor(self):
        return self.percent / 100.0

    def _remember_widget(self, widget):
        if widget in self._widgets:
            return
        saved = widget.property("_betterYoloWidgetScaleBaseline")
        if isinstance(saved, _ScaleBaseline):
            self._widgets[widget] = saved.values
            return
        state = {
            "minimum": widget.minimumSize(),
            "maximum": widget.maximumSize(),
            "style": widget.styleSheet(),
            "applied_style": widget.styleSheet(),
        }
        if isinstance(widget, (QAbstractButton, QAbstractItemView)):
            state["icon"] = widget.iconSize()
        if isinstance(widget, QListView):
            state["grid"] = widget.gridSize()
            state["spacing"] = widget.spacing()
        if isinstance(widget, QSplitter):
            state["handle"] = widget.handleWidth()
        if isinstance(widget, QTableView):
            state["row_height"] = widget.verticalHeader().defaultSectionSize()
        if isinstance(widget, QDialog):
            state["window_size"] = widget.size()
        self._widgets[widget] = state
        # Qt may destroy/recreate a Python wrapper while its C++ widget lives.
        # Weak caches alone then lose the original dimensions and inline style,
        # causing the already scaled values to be multiplied again next time.
        widget.setProperty("_betterYoloWidgetScaleBaseline", _ScaleBaseline(state))

    def _scale_layout(self, layout):
        if layout is None or sip.isdeleted(layout):
            return
        if layout not in self._layouts:
            saved = layout.property("_betterYoloLayoutScaleBaseline")
            if isinstance(saved, _ScaleBaseline):
                state = saved.values
            else:
                margins = layout.contentsMargins()
                state = {"margins": (margins.left(), margins.top(), margins.right(), margins.bottom())}
                if isinstance(layout, (QGridLayout, QFormLayout)):
                    state["horizontal"] = layout.horizontalSpacing()
                    state["vertical"] = layout.verticalSpacing()
                else:
                    state["spacing"] = layout.spacing()
                layout.setProperty("_betterYoloLayoutScaleBaseline", _ScaleBaseline(state))
            self._layouts[layout] = state
        state = self._layouts[layout]
        layout.setContentsMargins(*(round(n * self.factor) for n in state["margins"]))
        if state.get("spacing", -1) >= 0:
            layout.setSpacing(round(state["spacing"] * self.factor))
        if "horizontal" in state:
            if state["horizontal"] >= 0:
                layout.setHorizontalSpacing(round(state["horizontal"] * self.factor))
            if state["vertical"] >= 0:
                layout.setVerticalSpacing(round(state["vertical"] * self.factor))
        for i in range(layout.count()):
            self._scale_layout(layout.itemAt(i).layout())

    def _scaled_size(self, size):
        # QWIDGETSIZE_MAX is an unconstrained dimension, not a logical length.
        return QSize(*(n if n >= 16777215 else min(16777215, max(0, round(n * self.factor)))
                       for n in (size.width(), size.height())))

    def _apply_widget(self, widget):
        if sip.isdeleted(widget):
            return
        self._remember_widget(widget)
        state = self._widgets[widget]
        source = widget.styleSheet()
        if source != state["applied_style"]:
            state["style"] = source
        scaled = scale_stylesheet(state["style"], self.factor)
        if widget.styleSheet() != scaled:
            widget.setStyleSheet(scaled)
        state["applied_style"] = scaled
        if not isinstance(widget, QMainWindow):
            widget.setMinimumSize(QSize(0, 0))
            widget.setMaximumSize(self._scaled_size(state["maximum"]))
            widget.setMinimumSize(self._scaled_size(state["minimum"]))
        if "icon" in state:
            widget.setIconSize(self._scaled_size(state["icon"]))
        if "grid" in state:
            if state["grid"].isValid():
                widget.setGridSize(self._scaled_size(state["grid"]))
            widget.setSpacing(round(state["spacing"] * self.factor))
        if "handle" in state:
            widget.setHandleWidth(round(state["handle"] * self.factor))
        if "row_height" in state:
            widget.verticalHeader().setDefaultSectionSize(round(state["row_height"] * self.factor))
        if "window_size" in state:
            widget.resize(self._scaled_size(state["window_size"]))
        # Some existing pages keep their layout in an attribute named `layout`.
        self._scale_layout(QWidget.layout(widget))
        widget.updateGeometry()
        widget.update()

    def set_scale(self, percent):
        percent = max(self.MIN_PERCENT, min(self.MAX_PERCENT, int(percent)))
        self._applying = True
        try:
            widgets = [widget for widget in self.app.allWidgets() if self._is_managed(widget)]
            # Capture all metrics before changing the application stylesheet.
            for widget in widgets:
                self._remember_widget(widget)
            self.percent = percent
            self.app.setProperty("uiScaleFactor", self.factor)
            self.app.setStyleSheet(scale_stylesheet(self.stylesheet, self.factor))
            for widget in widgets:
                self._apply_widget(widget)
        finally:
            self._applying = False
        self.scale_changed.emit(percent)

    def _apply_new_widgets(self):
        self._pending = False
        self._applying = True
        try:
            for widget in self.app.allWidgets():
                if self._is_managed(widget) and widget not in self._widgets:
                    self._apply_widget(widget)
        finally:
            self._applying = False

    @staticmethod
    def _is_managed(widget):
        # Qt maintains these internal viewport/scrollbar geometries itself.
        return isinstance(widget, QWidget) and not widget.objectName().startswith("qt_")

    def eventFilter(self, obj, event):
        if self._applying or not self._is_managed(obj):
            return False
        if event.type() == QEvent.Type.StyleChange and obj in self._widgets:
            state = self._widgets[obj]
            if obj.styleSheet() != state["applied_style"]:
                self._applying = True
                try:
                    self._apply_widget(obj)
                finally:
                    self._applying = False
        elif event.type() == QEvent.Type.Show and obj not in self._widgets:
            self._applying = True
            try:
                self._apply_widget(obj)
                for child in obj.findChildren(QWidget):
                    if self._is_managed(child) and child not in self._widgets:
                        self._apply_widget(child)
            finally:
                self._applying = False
        elif event.type() == QEvent.Type.ChildAdded and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._apply_new_widgets)
        return False
