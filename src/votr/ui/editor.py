# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Voice editor: name, colour, Tone Hints, Tone Tags, Engine sliders."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QCompleter,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from votr.neural import NEURAL_ENGINE_ID
from votr.neural_engine import REFERENCE_CLIP_KEY
from votr.session import Session
from votr.ui.preview_panel import PreviewPanel


class VoiceEditor(QWidget):
    voice_saved = Signal()
    voice_deleted = Signal()

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self._loading = False
        self._build()
        self.reload_from_draft()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        preset_row = QHBoxLayout()
        self.preset_box = QComboBox()
        self.preset_box.addItem("Start from a preset…", "")
        for preset in self.session.presets:
            self.preset_box.addItem(preset.name, preset.name)
        self._refresh_preset_labels()
        self.preset_box.setToolTip(
            "Named slider recipes plus Tone Tags. If you already saved a Voice from "
            "a preset, Use preset opens that saved Voice; Fresh copy starts again "
            "from the bundled recipe."
        )
        use_preset = QPushButton("Use preset")
        use_preset.clicked.connect(lambda _=False: self._use_preset())
        fresh_copy = QPushButton("Fresh copy")
        fresh_copy.setToolTip("New unsaved draft from the bundled recipe.")
        fresh_copy.clicked.connect(self._fresh_preset)
        preset_row.addWidget(self.preset_box, 1)
        preset_row.addWidget(use_preset)
        preset_row.addWidget(fresh_copy)
        root.addLayout(preset_row)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._on_name)
        form.addRow("Name", self.name_edit)

        colour_row = QHBoxLayout()
        self.colour_button = QPushButton("Colour")
        self.colour_button.clicked.connect(self._pick_colour)
        colour_row.addWidget(self.colour_button)
        colour_row.addStretch()
        form.addRow("Colour", colour_row)

        self.hints_edit = QTextEdit()
        self.hints_edit.setPlaceholderText(
            "Tone Hints — notes for you, stored verbatim"
        )
        self.hints_edit.textChanged.connect(self._on_hints)
        form.addRow("Tone Hints", self.hints_edit)
        design_row = QHBoxLayout()
        design = QPushButton("Design from Tone Hints")
        design.setToolTip(
            "Reads the words above (deep, tiny, gravelly, ghostly, echoing…) and "
            "sets Tone Tags and sliders. Shapes how you sound; cannot change "
            "accent or make you a specific person."
        )
        design.clicked.connect(self.design_from_hints)
        self.design_status = QLabel()
        self.design_status.setWordWrap(True)
        self.design_status.setObjectName("design_status")
        design_row.addWidget(design)
        design_row.addWidget(self.design_status, 1)
        form.addRow("", design_row)
        root.addLayout(form)

        tags = QGroupBox("Tone Tags")
        tags_layout = QVBoxLayout(tags)
        tag_row = QHBoxLayout()
        self.tag_edit = QLineEdit()
        self.tag_edit.setPlaceholderText("Type a tag…")
        completer = QCompleter(sorted(self.session.macros))
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.tag_edit.setCompleter(completer)
        add = QPushButton("Add tag")
        add.clicked.connect(self._add_tag)
        self.tag_edit.returnPressed.connect(self._add_tag)
        tag_row.addWidget(self.tag_edit)
        tag_row.addWidget(add)
        tags_layout.addLayout(tag_row)
        self.tag_bar = QHBoxLayout()
        tags_layout.addLayout(self.tag_bar)
        root.addWidget(tags)

        sliders = QGroupBox("Sound")
        self.sliders_box = sliders
        slider_form = QFormLayout(sliders)
        for spec in self.session.engine.parameter_schema():
            row = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            value = QLabel()
            slider.valueChanged.connect(
                lambda pos, key=spec.key: self._on_slider(key, pos)
            )
            row.addWidget(slider)
            row.addWidget(value)
            slider_form.addRow(spec.label, row)
            self._sliders[spec.key] = slider
            self._slider_labels[spec.key] = value
        root.addWidget(sliders)

        self.preview_host = QGroupBox("Preview")
        preview_layout = QVBoxLayout(self.preview_host)

        self.preview = PreviewPanel(self.session)
        preview_layout.addWidget(self.preview)
        root.addWidget(self.preview_host)
        self.installEventFilter(self.preview)

        buttons = QHBoxLayout()
        save = QPushButton("Save")
        save.clicked.connect(self.save)
        save_new = QPushButton("Save as new")
        save_new.clicked.connect(self.save_as_new)
        delete = QPushButton("Delete")
        delete.clicked.connect(self.delete)
        buttons.addWidget(save)
        buttons.addWidget(save_new)
        buttons.addWidget(delete)
        buttons.addStretch()
        root.addLayout(buttons)
        root.addStretch()

    def reload_from_draft(self) -> None:
        self._loading = True
        self._refresh_preset_labels()
        draft = self.session.draft
        neural = draft.engine_id == NEURAL_ENGINE_ID
        self.sliders_box.setEnabled(not neural)
        if neural:
            clip = str(draft.params.get(REFERENCE_CLIP_KEY, "")) or "none"
            ready = (
                "ready"
                if self.session.engine_installed(NEURAL_ENGINE_ID)
                else (f"not ready — {self.session.neural.reason}")
            )
            self.sliders_box.setTitle(f"Sound — Neural Voice (Engine {ready})")
            self.sliders_box.setToolTip(f"Reference clip: {clip}")
        else:
            self.sliders_box.setTitle("Sound")
            self.sliders_box.setToolTip("")
        self.name_edit.setText(draft.name)
        self.hints_edit.setPlainText(draft.tone_hints)
        self._set_colour_button(draft.colour)
        self._refresh_tags()
        for spec in self.session.engine.parameter_schema():
            value = float(draft.params.get(spec.key, spec.default))
            span = spec.maximum - spec.minimum
            pos = int(round((value - spec.minimum) / span * 1000))
            self._sliders[spec.key].setValue(max(0, min(1000, pos)))
            self._slider_labels[spec.key].setText(f"{value:.2f}")
        self.session.apply_draft_to_engine()
        self._loading = False

    def _refresh_tags(self) -> None:
        while self.tag_bar.count():
            item = self.tag_bar.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for tag in self.session.draft.tone_tags:
            chip = QPushButton(f"{tag} ×")
            chip.clicked.connect(lambda _=False, name=tag: self._remove_tag(name))
            self.tag_bar.addWidget(chip)
        self.tag_bar.addStretch()

    def _set_colour_button(self, colour: str) -> None:
        self.colour_button.setStyleSheet(f"background: {colour};")
        self.colour_button.setText(colour)

    def _on_name(self, text: str) -> None:
        if not self._loading:
            self.session.draft.name = text

    def _on_hints(self) -> None:
        if not self._loading:
            self.session.draft.tone_hints = self.hints_edit.toPlainText()

    def _pick_colour(self) -> None:
        chosen = QColorDialog.getColor(QColor(self.session.draft.colour), self)
        if chosen.isValid():
            self.session.draft.colour = chosen.name()
            self._set_colour_button(self.session.draft.colour)

    def _add_tag(self) -> None:
        tag = self.tag_edit.text().strip().lower()
        self.tag_edit.clear()
        if not tag:
            return
        self.session.apply_tag(tag)
        self.reload_from_draft()

    def _refresh_preset_labels(self) -> None:
        for index in range(1, self.preset_box.count()):
            name = str(self.preset_box.itemData(index) or "")
            preset = next((p for p in self.session.presets if p.name == name), None)
            if preset is None:
                continue
            saved = self.session.saved_voice_for_preset(name)
            label = f"{name} — saved as “{saved.name}”" if saved else name
            self.preset_box.setItemText(index, label)
            tip = preset.description
            if saved:
                tip += " Use preset opens your saved Voice; Fresh copy starts over."
            self.preset_box.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)

    def _use_preset(self, *, fresh: bool = False) -> None:
        name = str(self.preset_box.currentData() or "")
        if not name or not self.confirm_discard():
            return
        voice = self.session.edit_from_preset(name, fresh=fresh)
        if voice is None:
            return
        if self.session.is_dirty():
            self.design_status.setText(
                f"New draft “{voice.name}” from the bundled “{name}” recipe — "
                "Preview, tune, then Save."
            )
        else:
            self.design_status.setText(
                f"Opened your saved “{voice.name}” (from “{name}”). Edits you Save "
                "stay; Fresh copy starts again from the bundled recipe."
            )
        self.reload_from_draft()
        self.preview.schedule_replay()

    def _fresh_preset(self) -> None:
        self._use_preset(fresh=True)

    def design_from_hints(self) -> None:
        prompt = self.hints_edit.toPlainText()
        if not prompt.strip():
            self.design_status.setText("Write a description in Tone Hints first.")
            return
        design = self.session.design_from_hints(prompt)
        parts = []
        if design.tone_tags:
            parts.append("Tags: " + ", ".join(design.tone_tags))
        if design.matched:
            parts.append("Heard: " + ", ".join(dict.fromkeys(design.matched)))
        parts.extend(design.notes)
        self.design_status.setText(" · ".join(parts))
        self.reload_from_draft()
        self.preview.schedule_replay()

    def _remove_tag(self, tag: str) -> None:
        self.session.draft.tone_tags = [
            item for item in self.session.draft.tone_tags if item != tag
        ]
        self._refresh_tags()

    def _on_slider(self, key: str, pos: int) -> None:
        spec = next(
            item for item in self.session.engine.parameter_schema() if item.key == key
        )
        value = spec.minimum + (spec.maximum - spec.minimum) * (pos / 1000.0)
        self._slider_labels[key].setText(f"{value:.2f}")
        if self._loading:
            return
        self.session.draft.params[key] = value
        self.session.engine.set_params({key: value})
        self.preview.schedule_replay()

    def save(self) -> None:
        self.session.save_draft()
        self.reload_from_draft()
        self.voice_saved.emit()

    def save_as_new(self) -> None:
        self.session.save_draft_as_new()
        self.reload_from_draft()
        self.voice_saved.emit()

    def delete(self) -> None:
        answer = QMessageBox.question(
            self,
            "Delete Voice",
            f"Delete “{self.session.draft.name}”?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.session.delete_draft()
        self.reload_from_draft()
        self.voice_deleted.emit()

    def confirm_discard(self) -> bool:
        if not self.session.is_dirty():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Discard unsaved changes to this Voice?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes
