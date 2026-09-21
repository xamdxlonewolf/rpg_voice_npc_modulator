# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Settings: devices, latency, hotkeys, Monitor, data folder, About."""

from __future__ import annotations

import sys
import threading

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QKeySequenceEdit,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from votr import WINDOW_TITLE, __version__
from votr.devices import find_cable_input, mic_devices
from votr.latency import format_latency_report
from votr.live import query_devices
from votr.neural import neural_status
from votr.neural_pack import (
    PACKS,
    DownloadCancelled,
    DownloadError,
    DownloadProgress,
    ModelPack,
    download_pack,
)
from votr.paths import license_path, notices_path
from votr.preview import speaker_devices
from votr.session import Session
from votr.store import default_data_dir
from votr.ui.mic_gain import MicGainSlider


class PackDownloadWorker(QThread):
    """Runs one pack download off the UI thread; cancel via ``cancel``."""

    progressed = Signal(object)
    finished_ok = Signal(str)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, pack: ModelPack, root, parent=None) -> None:
        super().__init__(parent)
        self._pack = pack
        self._root = root
        self.cancel = threading.Event()

    def run(self) -> None:  # noqa: D401 — QThread entry point
        try:
            download_pack(
                self._pack,
                self._root,
                progress=self.progressed.emit,
                cancel=self.cancel,
            )
        except DownloadCancelled:
            self.cancelled.emit(self._pack.pack_id)
        except DownloadError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit(self._pack.pack_id)


class SettingsDialog(QDialog):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Settings")
        self.setObjectName("settings_dialog")
        self.resize(520, 480)
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._devices_tab(), "Devices")
        tabs.addTab(self._latency_tab(), "Latency")
        tabs.addTab(self._general_tab(), "General")
        tabs.addTab(self._neural_tab(), "Neural")
        tabs.addTab(self._about_tab(), "About")
        root.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.clicked.connect(self._save)
        root.addWidget(buttons)

    def _devices_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.mic = QComboBox()
        self.speakers = QComboBox()
        devices = query_devices()
        mics = mic_devices(devices)
        if not mics:
            self.mic.addItem("No microphone", "")
        for device in mics:
            self.mic.addItem(str(device["name"]), str(device["name"]))
        speakers = speaker_devices(devices)
        if not speakers:
            self.speakers.addItem("No speakers", "")
        for device in speakers:
            self.speakers.addItem(str(device["name"]), str(device["name"]))
        if self.session.settings.mic_name:
            index = self.mic.findData(self.session.settings.mic_name)
            if index >= 0:
                self.mic.setCurrentIndex(index)
        if self.session.settings.speaker_name:
            index = self.speakers.findData(self.session.settings.speaker_name)
            if index >= 0:
                self.speakers.setCurrentIndex(index)
        cable = find_cable_input(
            devices, preferred=self.session.settings.cable_input_name
        )
        cable_name = cable["name"] if cable else "none — open Discord setup"
        layout.addRow("Microphone", self.mic)
        layout.addRow("Speakers", self.speakers)
        layout.addRow("Virtual Cable", QLabel(str(cable_name)))
        self.mic_gain = MicGainSlider(self.session)
        layout.addRow(self.mic_gain)
        return page

    def _latency_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.latency_report = QLabel(self._report_text())
        self.latency_report.setWordWrap(True)
        layout.addWidget(self.latency_report)
        rerun = QPushButton("Re-run test")
        rerun.clicked.connect(self._rerun)
        layout.addWidget(rerun)
        layout.addWidget(QLabel("Lower latency ↔ Cleaner sound"))
        self.latency_slider = QSlider(Qt.Orientation.Horizontal)
        self.latency_slider.setRange(0, 2)
        self.latency_slider.setValue(self.session.settings.latency_quality)
        self.latency_slider.valueChanged.connect(self._override)
        layout.addWidget(self.latency_slider)
        layout.addStretch()
        return page

    def _general_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.hotkey = QKeySequenceEdit(QKeySequence(self.session.settings.panic_hotkey))
        layout.addRow("Panic mute", self.hotkey)
        hint = QLabel(
            "In-app shortcut (available while Voice of the Realm is focused)."
        )
        hint.setWordWrap(True)
        layout.addRow("", hint)
        self.monitor = QCheckBox("Monitor Character Voice by default")
        self.monitor.setChecked(self.session.settings.monitor_on)
        layout.addRow(self.monitor)
        data = self.session.store.root.parent
        layout.addRow("Data folder", QLabel(str(data or default_data_dir())))
        open_voices = QPushButton("Open Voices folder")
        open_voices.clicked.connect(self._open_voices)
        layout.addRow(open_voices)
        return page

    def _neural_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        if self.session.neural.gpu_ok and not self.session.neural_loading:
            self.session.refresh_neural()
        runtime = self.session.neural
        self.neural_report = QLabel(neural_status(runtime.gpu, runtime))
        self.neural_report.setWordWrap(True)
        self.neural_report.setObjectName("neural_report")
        self.neural_report.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.neural_report)

        self.licence_ack = QCheckBox(
            "I have read the licences below and want to download model packs "
            "into my data folder (NVIDIA only; nothing is sent anywhere)."
        )
        self.licence_ack.setObjectName("licence_ack")
        self.licence_ack.toggled.connect(self._refresh_pack_buttons)
        layout.addWidget(self.licence_ack)

        self.pack_buttons: dict[str, QPushButton] = {}
        self.pack_labels: dict[str, QLabel] = {}
        for pack in PACKS:
            box = QGroupBox(f"{pack.title} — {pack.total_label}")
            box_layout = QVBoxLayout(box)
            summary = QLabel(pack.summary)
            summary.setWordWrap(True)
            box_layout.addWidget(summary)
            licences = QLabel(
                "Licences:\n• "
                + "\n• ".join(pack.licences)
                + ("\nNotes:\n• " + "\n• ".join(pack.caveats) if pack.caveats else "")
            )
            licences.setWordWrap(True)
            licences.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            box_layout.addWidget(licences)
            state = QLabel()
            state.setObjectName(f"pack_state_{pack.pack_id}")
            box_layout.addWidget(state)
            button = QPushButton()
            button.setObjectName(f"pack_button_{pack.pack_id}")
            button.clicked.connect(
                lambda _=False, pid=pack.pack_id: self._download(pid)
            )
            box_layout.addWidget(button)
            self.pack_buttons[pack.pack_id] = button
            self.pack_labels[pack.pack_id] = state
            layout.addWidget(box)

        self.pack_progress = QProgressBar()
        self.pack_progress.setRange(0, 1000)
        self.pack_progress.setVisible(False)
        layout.addWidget(self.pack_progress)
        self.pack_cancel = QPushButton("Cancel download")
        self.pack_cancel.setVisible(False)
        self.pack_cancel.clicked.connect(self._cancel_download)
        layout.addWidget(self.pack_cancel)
        self._worker: PackDownloadWorker | None = None
        self._refresh_pack_buttons()
        layout.addStretch()
        return page

    def _pack_state(self, pack: ModelPack) -> str:
        runtime = self.session.neural
        status = runtime.conversion if pack.pack_id == "xvc" else runtime.design
        if status.installed:
            return "Installed."
        if status.present:
            return (
                f"Partly downloaded ({status.bytes_present / 1024**3:.1f} GB present); "
                "Download resumes."
            )
        return "Not downloaded."

    def _refresh_pack_buttons(self) -> None:
        runtime = self.session.neural
        busy = self._worker is not None and self._worker.isRunning()
        for pack in PACKS:
            status = runtime.conversion if pack.pack_id == "xvc" else runtime.design
            button = self.pack_buttons[pack.pack_id]
            self.pack_labels[pack.pack_id].setText(self._pack_state(pack))
            if status.installed:
                button.setText("Installed")
                button.setEnabled(False)
                continue
            button.setText(f"Download {pack.total_label}")
            allowed = runtime.gpu_ok and self.licence_ack.isChecked() and not busy
            button.setEnabled(allowed)
            if not runtime.gpu_ok:
                button.setToolTip(
                    "Needs an NVIDIA GPU with enough memory; not offered here."
                )
            elif not self.licence_ack.isChecked():
                button.setToolTip("Tick the licence acknowledgement first.")
            else:
                button.setToolTip("")

    def _download(self, pack_id: str) -> None:
        pack = next((item for item in PACKS if item.pack_id == pack_id), None)
        if pack is None or (self._worker is not None and self._worker.isRunning()):
            return
        if not (self.session.neural.gpu_ok and self.licence_ack.isChecked()):
            return
        self._worker = PackDownloadWorker(pack, self.session.neural.root, self)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_download_done)
        self._worker.failed.connect(self._on_download_failed)
        self._worker.cancelled.connect(self._on_download_cancelled)
        self.pack_progress.setValue(0)
        self.pack_progress.setVisible(True)
        self.pack_cancel.setVisible(True)
        self._refresh_pack_buttons()
        self._worker.start()

    def _on_progress(self, progress: DownloadProgress) -> None:
        if progress.bytes_total:
            self.pack_progress.setValue(
                int(1000 * min(1.0, progress.bytes_done / progress.bytes_total))
            )
        self.pack_progress.setFormat(
            f"{progress.file_relpath} "
            f"({progress.file_index + 1}/{progress.file_count}) "
            f"{progress.bytes_done / 1024**2:.0f} MB"
        )

    def _finish_download(self, message: str) -> None:
        self.pack_progress.setVisible(False)
        self.pack_cancel.setVisible(False)
        self.session.refresh_neural()
        runtime = self.session.neural
        self.neural_report.setText(
            neural_status(runtime.gpu, runtime) + f"\n\n{message}"
        )
        self._refresh_pack_buttons()

    def _on_download_done(self, pack_id: str) -> None:
        self._finish_download(f"Download finished: {pack_id}.")

    def _on_download_failed(self, message: str) -> None:
        self._finish_download(f"Download failed: {message}")

    def _on_download_cancelled(self, pack_id: str) -> None:
        self._finish_download(
            f"Download cancelled: {pack_id}. Partial files kept; resume any time."
        )

    def _cancel_download(self) -> None:
        if self._worker is not None:
            self._worker.cancel.set()

    def _about_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(f"{WINDOW_TITLE} {__version__}"))
        layout.addWidget(QLabel("Licensed under GNU GPL-3.0-or-later."))
        if getattr(sys, "frozen", False):
            ship = (
                "This installed copy is the DSP app (Voices, Preview, Roleplay). "
                "CUDA / X-VC are not inside the installer. Neural still needs a "
                "source checkout: pip install -e \".[neural]\", then the opt-in "
                "packs on the Neural tab. See docs/neural-voice.md."
            )
        else:
            ship = (
                "The Windows installer ships DSP only. Neural needs this source "
                "tree, pip install -e \".[neural]\", and the opt-in packs on the "
                "Neural tab."
            )
        ship_label = QLabel(ship)
        ship_label.setWordWrap(True)
        ship_label.setObjectName("about_install_note")
        layout.addWidget(ship_label)
        notices = notices_path()
        license_file = license_path()
        if notices is not None:
            link = QPushButton("Third-party notices")
            link.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(notices)))
            )
            layout.addWidget(link)
        if license_file is not None:
            lic = QPushButton("LICENSE")
            lic.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(license_file)))
            )
            layout.addWidget(lic)
        layout.addStretch()
        return page

    def _report_text(self) -> str:
        settings = self.session.settings
        if not settings.latency_method:
            return "Latency Test has not been run. Re-run to measure this machine."
        from votr.latency import LatencyProbe

        probe = LatencyProbe(
            block_size=settings.block_size or 256,
            method=settings.latency_method,
            measured_ms=settings.latency_ms,
            engine_ms=settings.latency_engine_ms,
            device_ms=settings.latency_device_ms,
            total_ms=settings.latency_ms or settings.latency_engine_ms,
            underruns=None,
            blocked=settings.latency_blocked or None,
        )
        return format_latency_report(probe)

    def _rerun(self) -> None:
        duration = 10.0 if query_devices() else 0.0
        probe = self.session.run_latency_test(duration_s=duration)
        self.latency_report.setText(format_latency_report(probe))
        self.latency_slider.blockSignals(True)
        self.latency_slider.setValue(self.session.settings.latency_quality)
        self.latency_slider.blockSignals(False)

    def _override(self, quality: int) -> None:
        self.session.apply_latency_quality(quality)

    def _open_voices(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session.store.root)))

    def _save(self) -> None:
        self.session.settings.mic_name = str(self.mic.currentData() or "")
        self.session.settings.speaker_name = str(self.speakers.currentData() or "")
        cable = find_cable_input()
        if cable is not None:
            self.session.settings.cable_input_name = str(cable["name"])
        self.session.settings.monitor_on = self.monitor.isChecked()
        sequence = self.hotkey.keySequence().toString()
        if sequence:
            self.session.settings.panic_hotkey = sequence
        self.session.save_settings()
