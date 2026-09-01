"""Tkinter desktop application for orchestrating FreeTune4D."""

from __future__ import annotations

from datetime import datetime
import ctypes
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from .backend import BackendError, FreeTune4DBackend, OutputSummary, PipelineConfig
from .controller import WorkflowController, WorkflowState
from .devices import DeviceInfo
from .dialogs import choose_directory
from .typography import TYPOGRAPHY, configure_named_fonts


class FreeTune4DApp(tk.Tk):
    POLL_MS = 100
    WINDOW_SIZE = (1280, 840)
    MINIMUM_SIZE = (1100, 720)
    COLUMN_WEIGHTS = (65, 35)
    FONT_SIZES = {
        "title": TYPOGRAPHY.TITLE_PX,
        "section": TYPOGRAPHY.SECTION_PX,
        "normal": TYPOGRAPHY.BODY_PX,
        "log": TYPOGRAPHY.LOG_PX,
    }

    def __init__(self, backend: FreeTune4DBackend | None = None):
        super().__init__()
        self.backend = backend or FreeTune4DBackend()
        self.controller = WorkflowController()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_validation_errors: list[str] = []
        self.output_summary: OutputSummary | None = None
        self.last_error_details = ""
        self.active_device: DeviceInfo | None = None
        self.last_directory = Path.home()

        self.title("FreeTune4D — UQ 4D-MRI Motion Reconstruction")
        self.geometry(f"{self.WINDOW_SIZE[0]}x{self.WINDOW_SIZE[1]}")
        self.minsize(*self.MINIMUM_SIZE)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._create_variables()
        self._configure_style()
        self._build_ui()
        self.after(self.POLL_MS, self._poll_events)
        self._validate_form(log_success=False)
        if os.environ.get("FREETUNE4D_DEBUG_UI") == "1":
            self.after_idle(self._print_ui_diagnostics)

    def _create_variables(self) -> None:
        repo = self.backend.repo_root
        self.dynamic_var = tk.StringVar()
        self.static_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.modality_var = tk.StringVar(value="T2-weighted")
        self.coarse_var = tk.StringVar(value=str(repo / "models" / "coarse.h5"))
        self.fine_var = tk.StringVar(value=str(repo / "models" / "fine.h5"))
        self.reference_var = tk.StringVar()
        self.phase_var = tk.StringVar(value="5")
        self.cuda_diagnostics_var = tk.BooleanVar(value=False)
        self.compute_device_var = tk.StringVar(value="Auto — Recommended")
        self.overall_status_var = tk.StringVar(value="Select valid inputs to begin.")
        self.input_status_var = tk.StringVar(value="Waiting")
        self.preprocess_status_var = tk.StringVar(value="Waiting")
        self.reconstruct_status_var = tk.StringVar(value="Waiting")
        self.output_status_var = tk.StringVar(value="Waiting")
        for variable in (
            self.dynamic_var,
            self.static_var,
            self.output_var,
            self.modality_var,
            self.coarse_var,
            self.fine_var,
            self.reference_var,
            self.phase_var,
            self.cuda_diagnostics_var,
            self.compute_device_var,
        ):
            variable.trace_add("write", self._on_config_changed)

    def _configure_style(self) -> None:
        self.fonts = configure_named_fonts(self)
        self.option_add("*Font", "FreeTune4DBody")
        self.option_add("*Text.Font", "FreeTune4DBody")
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        body_linespace = self.fonts["body"].metrics("linespace")
        control_y_padding = max(1, (TYPOGRAPHY.CONTROL_HEIGHT_PX - body_linespace) // 2)
        primary_y_padding = max(1, (TYPOGRAPHY.PRIMARY_HEIGHT_PX - body_linespace) // 2)
        style.configure("TLabel", font="FreeTune4DBody")
        style.configure("TEntry", font="FreeTune4DBody", padding=(8, control_y_padding))
        style.configure("TCombobox", font="FreeTune4DBody", padding=(8, control_y_padding))
        style.configure("TSpinbox", font="FreeTune4DBody", padding=(8, control_y_padding))
        style.configure("TButton", font="FreeTune4DMedium", padding=(12, control_y_padding))
        style.configure("Header.TLabel", font="FreeTune4DTitle", foreground="#17324d")
        style.configure("Subheader.TLabel", font="FreeTune4DBody", foreground="#526577")
        style.configure("Section.TLabel", font="FreeTune4DSection", foreground="#17324d")
        style.configure("Section.TLabelframe.Label", font="FreeTune4DSection", foreground="#17324d")
        style.configure("Section.TLabelframe", borderwidth=1, relief="solid")
        style.configure("Primary.TButton", font="FreeTune4DSemibold", padding=(16, primary_y_padding))
        style.configure("StatusName.TLabel", font="FreeTune4DMedium")
        style.configure("Status.TLabel", font="FreeTune4DMedium", padding=(9, 6), background="#edf2f6")
        style.configure("Operation.TLabel", font="FreeTune4DMedium", foreground="#17324d")
        style.configure("Link.TButton", font="FreeTune4DMedium", padding=(10, control_y_padding))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, name="central_widget", padding=(20, 16))
        self.central_widget = outer
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.page_title = ttk.Label(header, name="page_title", text="FreeTune4D", style="Header.TLabel")
        self.page_title.pack(anchor="w")
        ttk.Label(header, text="UQ 4D-MRI Motion Reconstruction", style="Subheader.TLabel").pack(anchor="w")

        self.vertical_panes = ttk.Panedwindow(outer, orient="vertical")
        self.vertical_panes.grid(row=1, column=0, sticky="nsew")
        upper = ttk.Frame(self.vertical_panes)
        log_frame = ttk.Labelframe(self.vertical_panes, name="runtime_log_section", text="Runtime Log", padding=8, style="Section.TLabelframe")
        self.log_frame = log_frame
        # Let the configuration/workflow area request its natural height; the
        # Runtime Log receives the remaining flexible vertical space.
        self.vertical_panes.add(upper, weight=0)
        self.vertical_panes.add(log_frame, weight=1)

        upper.columnconfigure(0, weight=self.COLUMN_WEIGHTS[0], uniform="main-columns", minsize=620)
        upper.columnconfigure(1, weight=self.COLUMN_WEIGHTS[1], uniform="main-columns", minsize=390)
        upper.rowconfigure(0, weight=1)
        left = ttk.Frame(upper, padding=(0, 0, 9, 0))
        right = ttk.Frame(upper, padding=(9, 0, 0, 0))
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        input_frame = ttk.Labelframe(left, name="input_data_section", text="Input Data", padding=14, style="Section.TLabelframe")
        self.input_frame = input_frame
        input_frame.pack(fill="x", pady=(0, 14))
        input_frame.columnconfigure(0, weight=1)
        self.dynamic_entry, self.dynamic_browse = self._path_row(
            input_frame, 0, "Dynamic LQ 4D-MRI DICOM",
            "Respiratory-resolved low-quality 4D-MRI containing motion information.",
            self.dynamic_var, lambda: self._browse_directory(self.dynamic_var),
        )
        self.static_entry, self.static_browse = self._path_row(
            input_frame, 3, "Static UQ 3D-MRI DICOM",
            "High-quality 3D MRI providing anatomical prior information.",
            self.static_var, lambda: self._browse_directory(self.static_var),
        )
        self.modality_label = ttk.Label(input_frame, name="mri_modality_label", text="MRI Modality", style="StatusName.TLabel")
        self.modality_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 4))
        self.modality_combo = ttk.Combobox(input_frame, textvariable=self.modality_var, values=("T2-weighted", "T1-weighted"), state="readonly", width=22)
        self.modality_combo.grid(row=7, column=0, sticky="w", pady=(0, 5))
        self.modality_note = ttk.Label(input_frame, text="T2 backend available; T1 backend is not present.", foreground="#7a5a00")
        self.modality_note.grid(row=8, column=0, columnspan=2, sticky="w")

        ttk.Label(input_frame, text="Compute Device", style="StatusName.TLabel").grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(10, 4)
        )
        device_row = ttk.Frame(input_frame)
        device_row.grid(row=10, column=0, columnspan=2, sticky="ew")
        device_row.columnconfigure(0, weight=1)
        self.compute_device_combo = ttk.Combobox(
            device_row, textvariable=self.compute_device_var, state="readonly"
        )
        self.compute_device_combo.grid(row=0, column=0, sticky="ew", padx=(0, 9))
        self.refresh_devices_button = ttk.Button(device_row, text="Refresh", command=self._refresh_device_list)
        self.refresh_devices_button.grid(row=0, column=1)
        self.device_status = ttk.Label(input_frame, foreground="#526577", wraplength=590)
        self.device_status.grid(row=11, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.device_display_to_key: dict[str, str] = {}
        self._refresh_device_list(log_devices=False)

        output_frame = ttk.Labelframe(left, text="Output", padding=14, style="Section.TLabelframe")
        output_frame.pack(fill="x", pady=(0, 14))
        output_frame.columnconfigure(0, weight=1)
        self.output_entry, self.output_browse = self._path_row(
            output_frame, 0, "Output Directory", "Creates preprocessing, reconstructed and QC below this root.",
            self.output_var, lambda: self._browse_directory(self.output_var),
        )

        self.advanced = ttk.Frame(left)
        self.advanced.pack(fill="x")
        self.advanced_visible = tk.BooleanVar(value=False)
        self.advanced_toggle = ttk.Button(self.advanced, name="advanced_settings", text="Advanced Settings ▸", command=self._toggle_advanced)
        self.advanced_toggle.grid(row=0, column=0, sticky="ew")
        self.advanced.columnconfigure(0, weight=1)
        self.advanced_body = ttk.Labelframe(self.advanced, text="Backend Settings", padding=12, style="Section.TLabelframe")
        self._file_row(self.advanced_body, 0, "Coarse model (coarse.h5)", self.coarse_var, lambda: self._browse_file(self.coarse_var, (("H5 model", "*.h5"),)))
        self._file_row(self.advanced_body, 1, "Fine model (fine.h5)", self.fine_var, lambda: self._browse_file(self.fine_var, (("H5 model", "*.h5"),)))
        self._file_row(self.advanced_body, 2, "Reference DICOM (optional)", self.reference_var, lambda: self._browse_file(self.reference_var, (("DICOM", "*.dcm"), ("All files", "*"))))
        ttk.Label(self.advanced_body, text="Respiratory phases").grid(row=3, column=0, sticky="w", pady=7)
        self.phase_spin = ttk.Spinbox(self.advanced_body, from_=1, to=65, textvariable=self.phase_var, width=8)
        self.phase_spin.grid(row=3, column=1, sticky="w", padx=8, pady=7)
        self.cuda_diagnostics_check = ttk.Checkbutton(
            self.advanced_body,
            text="CUDA diagnostic mode (CUDA_LAUNCH_BLOCKING=1; slower)",
            variable=self.cuda_diagnostics_var,
        )
        self.cuda_diagnostics_check.grid(row=4, column=0, columnspan=3, sticky="w", pady=(7, 2))
        self.advanced_controls.append(self.cuda_diagnostics_check)
        self.advanced_body.columnconfigure(1, weight=1)

        workflow = ttk.Labelframe(right, name="workflow_section", text="Reconstruction Workflow", padding=16, style="Section.TLabelframe")
        self.workflow_frame = workflow
        workflow.pack(fill="x")
        status_grid = ttk.Frame(workflow)
        status_grid.pack(fill="x")
        self.workflow_name_labels = []
        self.workflow_status_labels = []
        for row, (number, title, variable) in enumerate((
            ("1", "Input", self.input_status_var),
            ("2", "Preprocessing", self.preprocess_status_var),
            ("3", "Motion Reconstruction", self.reconstruct_status_var),
            ("4", "Output", self.output_status_var),
        )):
            ttk.Label(status_grid, text=number, style="StatusName.TLabel", width=3).grid(row=row, column=0, sticky="nw", pady=8)
            item = ttk.Frame(status_grid)
            item.grid(row=row, column=1, sticky="ew", pady=8)
            name_label = ttk.Label(item, text=title, style="StatusName.TLabel")
            name_label.pack(anchor="w")
            status_label = ttk.Label(item, textvariable=variable, style="Status.TLabel")
            status_label.pack(anchor="w", pady=(3, 0))
            self.workflow_name_labels.append(name_label)
            self.workflow_status_labels.append(status_label)
        status_grid.columnconfigure(1, weight=1)

        self.preprocess_button = ttk.Button(workflow, text="Preprocessing", style="Primary.TButton", command=self._start_preprocessing)
        self.preprocess_button.pack(fill="x", pady=(14, 8))
        self.reconstruct_button = ttk.Button(workflow, text="Motion Reconstruction", style="Primary.TButton", command=self._start_reconstruction)
        self.reconstruct_button.pack(fill="x", pady=(0, 16))

        operation = ttk.Labelframe(workflow, text="Current Operation", padding=12, style="Section.TLabelframe")
        self.operation_frame = operation
        operation.pack(fill="x", pady=(0, 14))
        ttk.Label(operation, textvariable=self.overall_status_var, style="Operation.TLabel", wraplength=360).pack(anchor="w", fill="x")
        self.activity = ttk.Progressbar(operation, mode="indeterminate")
        self.activity.pack(fill="x", pady=(10, 0))

        results = ttk.Labelframe(workflow, text="Output Summary", padding=12, style="Section.TLabelframe")
        self.results_frame = results
        results.pack(fill="both", expand=True)
        self.results_text = tk.Text(results, height=8, wrap="word", state="disabled", background="#f7f9fb", relief="flat")
        self.results_text.pack(fill="both", expand=True)

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(log_toolbar, text="Clear Log", style="Link.TButton", command=self._clear_log).pack(side="right")
        self.copy_error_button = ttk.Button(log_toolbar, text="Copy Error", style="Link.TButton", command=self._copy_error)
        self.copy_error_button.pack(side="right", padx=(0, 8))
        self.log_text = tk.Text(
            log_frame, height=12, wrap="none", state="disabled",
            background="#101820", foreground="#dce5ec", insertbackground="white",
            selectbackground="#35566f", font="TkFixedFont", undo=False,
        )
        log_scroll_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll_x = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=log_scroll_y.set, xscrollcommand=log_scroll_x.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll_y.pack(side="right", fill="y")
        log_scroll_x.pack(side="bottom", fill="x")

        actions = ttk.Frame(outer)
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.open_output_button = ttk.Button(actions, text="Open Output Folder", command=lambda: self._open_folder(self._config().output_root))
        self.open_reconstructed_button = ttk.Button(actions, text="Open Reconstructed Folder", command=lambda: self._open_folder(self.backend.output_paths(self._config())[1]))
        self.open_qc_button = ttk.Button(actions, text="Open QC Folder", command=lambda: self._open_folder(self.backend.output_paths(self._config())[2]))
        for button in (self.open_output_button, self.open_reconstructed_button, self.open_qc_button):
            button.pack(side="left", padx=(0, 10))
        self._update_controls()

    def _path_row(self, parent, row, label, description, variable, browse_command):
        label_widget = ttk.Label(parent, text=label, style="StatusName.TLabel")
        label_widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8 if row else 0, 2))
        ttk.Label(parent, text=description, foreground="#657786", wraplength=590).grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row + 2, column=0, sticky="ew", padx=(0, 9), pady=(0, 5))
        button = ttk.Button(parent, text="Browse...", width=10, command=browse_command)
        button.grid(row=row + 2, column=1, sticky="e", pady=(0, 5))
        if label.startswith("Dynamic"):
            self.dynamic_label = label_widget
        elif label.startswith("Static"):
            self.static_label = label_widget
        elif label == "Output Directory":
            self.output_label = label_widget
        return entry, button

    def _file_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=7)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=7)
        button = ttk.Button(parent, text="Browse...", width=10, command=command)
        button.grid(row=row, column=2, pady=7)
        if not hasattr(self, "advanced_controls"):
            self.advanced_controls = []
        self.advanced_controls.extend((entry, button))

    def _toggle_advanced(self) -> None:
        if self.advanced_visible.get():
            self.advanced_visible.set(False)
            self.advanced_body.grid_remove()
            self.advanced_toggle.configure(text="Advanced Settings ▸")
        else:
            self.advanced_visible.set(True)
            self.advanced_body.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            self.advanced_toggle.configure(text="Advanced Settings ▾")

    def _browse_directory(self, variable: tk.StringVar) -> None:
        initial = Path(variable.get()) if variable.get().strip() else self.last_directory
        selected = choose_directory(self, initial)
        if selected:
            self.last_directory = selected
            variable.set(str(selected))

    def _browse_file(self, variable: tk.StringVar, filetypes) -> None:
        selected = filedialog.askopenfilename(initialdir=str(Path(variable.get()).parent) if variable.get() else str(Path.home()), filetypes=filetypes)
        if selected:
            variable.set(selected)

    def _config(self) -> PipelineConfig:
        modality = "T1" if self.modality_var.get().startswith("T1") else "T2"
        reference = Path(self.reference_var.get()) if self.reference_var.get().strip() else None
        try:
            phase_count = int(self.phase_var.get())
        except ValueError:
            phase_count = 0
        return PipelineConfig(
            Path(self.dynamic_var.get() or "."),
            Path(self.static_var.get() or "."),
            Path(self.output_var.get() or "."),
            modality,
            Path(self.coarse_var.get() or "."),
            Path(self.fine_var.get() or "."),
            reference,
            phase_count,
            self.cuda_diagnostics_var.get(),
            self.device_display_to_key.get(self.compute_device_var.get(), "auto"),
        )

    def _refresh_device_list(self, log_devices: bool = True) -> None:
        previous_key = self.device_display_to_key.get(self.compute_device_var.get(), "auto") if hasattr(self, "device_display_to_key") else "auto"
        devices = self.backend.refresh_devices()
        choices = ["Auto — Recommended"]
        mapping = {"Auto — Recommended": "auto"}
        for device in devices:
            choices.append(device.display_name)
            mapping[device.display_name] = device.key
        self.device_display_to_key = mapping
        selected_display = next((display for display, key in mapping.items() if key == previous_key), "Auto — Recommended")
        self.compute_device_combo.configure(values=choices)
        self.compute_device_var.set(selected_display)
        selected = self.backend.resolve_device(self._config())
        if selected and selected.device_type == "cpu":
            self.device_status.configure(
                text="Current backend requires CUDA. CPU execution has not been validated."
            )
        elif selected:
            suffix = " — Low available VRAM" if selected.low_memory else ""
            prefix = "Auto will use " if self._config().compute_device == "auto" else "Selected "
            self.device_status.configure(text=f"{prefix}GPU {selected.physical_index}: {selected.free_gib:.2f}/{selected.total_gib:.2f} GiB free{suffix}")
        else:
            self.device_status.configure(text="No compatible CUDA GPU detected. CPU is unavailable in the current backend.")
        if log_devices and hasattr(self, "log_text"):
            gpu_count = sum(device.device_type == "cuda" for device in devices)
            self._log(f"[DEVICE] Refreshed CUDA devices: {gpu_count} detected")

    def _preflight_device(self, config: PipelineConfig, operation: str) -> bool:
        self._refresh_device_list(log_devices=False)
        config = self._config().normalized()
        device = self.backend.resolve_device(config)
        if device is None or not device.available or not device.supported:
            messagebox.showerror("Compute device unavailable", self.backend.device_error(config))
            return False
        self.active_device = device
        self._log(f"[DEVICE] Pre-flight for {operation}: {device.display_name}")
        if device.low_memory:
            return messagebox.askyesno(
                "Low GPU memory",
                f"GPU {device.physical_index} currently has only {device.free_gib:.2f} GiB free "
                f"out of {device.total_gib:.2f} GiB.\n\n{operation.title()} may fail with CUDA out-of-memory. "
                "Choose another GPU or free GPU memory.\n\nContinue anyway?",
            )
        return True

    def _on_config_changed(self, *_args) -> None:
        if hasattr(self, "preprocess_button") and not self.controller.busy:
            self._validate_form(log_success=False)

    def _validate_form(self, log_success: bool = True) -> bool:
        config = self._config()
        errors = self.backend.validate_inputs(config)
        device_error = self.backend.device_error(config)
        if device_error:
            errors.append(device_error)
        previous_signature = self.controller.config_signature
        self.controller.inputs_validated(config, errors)
        if previous_signature and previous_signature != self.controller.config_signature:
            self.output_summary = None
            self._set_results("Configuration changed. Previous preprocessing is no longer valid.")
        self.last_validation_errors = errors
        if errors:
            self.overall_status_var.set(errors[0])
        else:
            self.overall_status_var.set("Inputs and backend resources are ready.")
            if log_success:
                self._log("Input validation completed.")
        self._update_controls()
        return not errors

    def _start_preprocessing(self) -> None:
        if not self._validate_form():
            messagebox.showerror("Invalid configuration", "\n".join(self.last_validation_errors))
            return
        config = self._config().normalized()
        if not self._preflight_device(config, "preprocessing"):
            return
        config = self._config().normalized()
        if self.backend.has_existing_outputs(config):
            replace = messagebox.askyesno("Existing output detected", "Managed output folders contain data. Re-run and replace these outputs?\n\nNo data will be deleted unless you choose Yes.")
            if not replace:
                return
            self.backend.clear_managed_outputs(config)
        self.controller.start_preprocessing()
        self._log_config(config)
        self._log("Starting preprocessing...")
        self.overall_status_var.set("Preprocessing...")
        self._start_worker("preprocessing", lambda: self.backend.run_preprocessing(config, self._queue_log))

    def _start_reconstruction(self) -> None:
        config = self._config().normalized()
        if not self.controller.can_reconstruct or not self.backend.validate_preprocessing_output(config):
            messagebox.showerror("Preprocessing required", "Successful preprocessing for the current case is required.")
            self._validate_form(log_success=False)
            return
        if not self._preflight_device(config, "motion reconstruction"):
            return
        config = self._config().normalized()
        reconstructed = self.backend.output_paths(config)[1]
        if reconstructed.exists() and any(reconstructed.iterdir()):
            if not messagebox.askyesno("Existing reconstruction", "Reconstructed output already exists. Replace it and continue?"):
                return
            shutil.rmtree(reconstructed)
            reconstruction_qc = self.backend.output_paths(config)[2] / "reconstruction"
            if reconstruction_qc.exists():
                shutil.rmtree(reconstruction_qc)
        self.controller.start_reconstruction()
        self._log("Starting motion reconstruction...")
        self.overall_status_var.set("Motion Reconstruction...")
        self._start_worker("reconstruction", lambda: self.backend.run_motion_reconstruction(config, self._queue_log))

    def _start_worker(self, stage: str, operation) -> None:
        self._update_controls()
        self.activity.start(12)

        def run():
            try:
                result = operation()
            except Exception as exc:
                self.events.put(("failure", (stage, exc, traceback.format_exc())))
            else:
                self.events.put(("success", (stage, result)))

        self.worker = threading.Thread(target=run, name=f"FreeTune4D-{stage}", daemon=True)
        self.worker.start()

    def _queue_log(self, message: str) -> None:
        self.events.put(("log", message))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self._handle_backend_log(str(payload))
                elif event == "success":
                    self._operation_succeeded(*payload)
                elif event == "failure":
                    self._operation_failed(*payload)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._poll_events)

    def _handle_backend_log(self, message: str) -> None:
        self._log(message)
        if self.controller.state == WorkflowState.PREPROCESSING:
            batch = re.search(r"computing batch (\d+)/(\d+)", message)
            if batch:
                self.overall_status_var.set(f"Preprocessing — similarity batch {batch.group(1)} / {batch.group(2)}")
        elif self.controller.state == WorkflowState.RECONSTRUCTING:
            frame = re.search(r"(?:Frame|frame)[ _]?(\d+)", message)
            if frame:
                self.overall_status_var.set(f"Reconstructing respiratory phase {int(frame.group(1)) + 1}")

    def _operation_succeeded(self, stage: str, result: object) -> None:
        self.activity.stop()
        if stage == "preprocessing":
            self.controller.preprocessing_succeeded()
            self.overall_status_var.set("Preprocessing completed")
            self._log("Preprocessing completed. Motion Reconstruction enabled.")
        else:
            self.controller.reconstruction_succeeded()
            self.output_summary = result if isinstance(result, OutputSummary) else None
            self.overall_status_var.set("Reconstruction completed")
            self._log("Motion reconstruction process completed.")
            self._show_output_summary()
        self._update_controls()

    def _operation_failed(self, stage: str, exc: Exception, details: str) -> None:
        self.activity.stop()
        self.controller.operation_failed(stage)
        label = "Preprocessing" if stage == "preprocessing" else "Motion reconstruction"
        if isinstance(exc, BackendError):
            concise = self._backend_error_message(exc)
            self.last_error_details = exc.details + "\n\n--- GUI worker traceback ---\n" + details
        else:
            concise = str(exc)
            self.last_error_details = details
        self.overall_status_var.set(f"{label} failed\n{concise}\nSee Runtime Log for details.")
        self._log(f"ERROR: {label} failed: {concise}")
        self._log(self.last_error_details)
        self._show_error_dialog(label, concise)
        self._update_controls()

    def _update_controls(self) -> None:
        busy = self.controller.busy
        editable_state = "disabled" if busy else "normal"
        for entry in (self.dynamic_entry, self.static_entry, self.output_entry):
            entry.configure(state=editable_state)
        for button in (self.dynamic_browse, self.static_browse, self.output_browse):
            button.configure(state=editable_state)
        self.modality_combo.configure(state="disabled" if busy else "readonly")
        self.compute_device_combo.configure(state="disabled" if busy else "readonly")
        self.refresh_devices_button.configure(state="disabled" if busy else "normal")
        self.phase_spin.configure(state=editable_state)
        for control in getattr(self, "advanced_controls", []):
            control.configure(state=editable_state)
        self.preprocess_button.configure(state="normal" if self.controller.can_preprocess else "disabled")
        self.reconstruct_button.configure(state="normal" if self.controller.can_reconstruct else "disabled")

        state = self.controller.state
        self.input_status_var.set("Ready" if state != WorkflowState.INVALID else "Waiting")
        self.preprocess_status_var.set({
            WorkflowState.PREPROCESSING: "Running...",
            WorkflowState.PREPROCESSED: "Completed",
            WorkflowState.RECONSTRUCTING: "Completed",
            WorkflowState.COMPLETED: "Completed",
        }.get(state, "Failed" if state == WorkflowState.FAILED and self.controller.failure_stage == "preprocessing" else "Waiting"))
        reconstruct_failed = state == WorkflowState.FAILED and self.controller.failure_stage == "reconstruction"
        self.reconstruct_status_var.set({WorkflowState.RECONSTRUCTING: "Running...", WorkflowState.COMPLETED: "Completed"}.get(state, "Failed" if reconstruct_failed else "Waiting"))
        self.output_status_var.set("Completed" if state == WorkflowState.COMPLETED else "Waiting")
        config = self._config()
        paths = self.backend.output_paths(config)
        self.open_output_button.configure(state="normal" if config.output_root.is_dir() else "disabled")
        self.open_reconstructed_button.configure(state="normal" if paths[1].is_dir() else "disabled")
        self.open_qc_button.configure(state="normal" if paths[2].is_dir() else "disabled")

    def _show_output_summary(self) -> None:
        summary = self.output_summary
        if not summary:
            self._set_results("⚠ Reconstruction completed, but output validation did not return a summary.")
            return
        lines = [
            ("✓" if summary.uq_complete else "⚠") + f" Reconstructed UQ 4D-MRI DICOM: {summary.uq_dicom_count} files, {summary.respiratory_phases} phases detected",
            ("✓" if summary.lq_complete else "⚠") + f" LQ 4D-MRI DICOM backup/output: {summary.lq_dicom_count} files",
            ("✓" if summary.qc_complete else "⚠") + f" QC visualization: {summary.qc_file_count} files",
            f"Output: {summary.reconstructed_dir}",
            f"QC: {summary.qc_dir}",
        ]
        self._set_results("\n".join(lines))

    def _set_results(self, text: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    def _log_config(self, config: PipelineConfig) -> None:
        self._log(f"Dynamic input: {config.dynamic_dicom_dir}")
        self._log(f"Static input: {config.static_dicom_dir}")
        self._log(f"Modality: {config.modality}")
        self._log(f"Output root: {config.output_root}")
        self._log(f"Respiratory phases: {config.phase_count}")
        self._log(f"CUDA diagnostic mode: {'enabled' if config.cuda_diagnostics else 'disabled'}")
        self._log(f"Compute device mode: {config.compute_device}")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        at_bottom = self.log_text.yview()[1] >= 0.98
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        if at_bottom:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _backend_error_message(self, error: BackendError) -> str:
        if error.kind == "cuda_oom":
            device = self.active_device
            selected = device.display_name if device else "the selected GPU"
            memory = (
                f"\nTotal GPU memory: {device.total_gib:.2f} GiB"
                f"\nFree memory before launch: approximately {device.free_gib:.2f} GiB"
                if device else ""
            )
            return (
                f"GPU memory exhausted during {error.operation}.\nSelected device: {selected}{memory}\n\n"
                "Try selecting another GPU, closing other GPU workloads, or retrying after GPU memory becomes available."
            )
        if error.kind == "cuda":
            if "device-side assert" in (error.root_message + error.stderr_tail).lower():
                return f"CUDA device-side assertion detected. {error.root_message}"
            return f"CUDA error occurred during backend {error.operation}. {error.root_message}"
        labels = {
            "model_loading": "Model loading failed",
            "missing_dependency": "A required backend dependency is unavailable",
            "invalid_input": "Backend input validation failed",
            "output_validation": "Expected backend output was not generated",
        }
        prefix = labels.get(error.kind, "Backend operation failed")
        return f"{prefix}. {error.root_message}"

    def _show_error_dialog(self, label: str, concise: str) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Operation failed")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(True, False)
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"{label} failed", style="Section.TLabel", foreground="#a32929").pack(anchor="w")
        ttk.Label(body, text=concise, wraplength=600).pack(anchor="w", fill="x", pady=(10, 4))
        ttk.Label(body, text="Detailed traceback is available in Runtime Log.", foreground="#526577").pack(anchor="w")
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(16, 0))
        ttk.Button(actions, text="Copy Error Details", command=self._copy_error).pack(side="left")
        ttk.Button(actions, text="Close", command=dialog.destroy).pack(side="right")

    def _copy_error(self) -> None:
        if not self.last_error_details:
            return
        self.clipboard_clear()
        self.clipboard_append(self.last_error_details)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _open_folder(path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.is_dir():
            messagebox.showwarning("Folder unavailable", f"Directory does not exist:\n{path}")
            return
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _effective_font(self, widget: tk.Misc, labelframe_title: bool = False) -> tkfont.Font:
        """Return the font Tk actually resolves for a widget after styling."""
        font_spec = ""
        if not labelframe_title:
            try:
                font_spec = str(widget.cget("font"))
            except tk.TclError:
                pass
        if not font_spec and isinstance(widget, ttk.Widget):
            style = ttk.Style(self)
            style_name = str(widget.cget("style")) or widget.winfo_class()
            if labelframe_title:
                style_name += ".Label" if style_name != "TLabelframe" else ".Label"
            font_spec = str(style.lookup(style_name, "font"))
        return tkfont.Font(root=self, font=font_spec or "TkDefaultFont")

    def _print_ui_diagnostics(self) -> None:
        """Print effective post-layout typography when FREETUNE4D_DEBUG_UI=1."""
        self.update_idletasks()
        logical_dpi = float(self.winfo_fpixels("1i"))
        scaling = float(self.tk.call("tk", "scaling"))
        print("[FONT DEBUG]")
        print(
            f"Screen: logicalDPI={logical_dpi:.2f}, tkScaling={scaling:.4f}, "
            f"devicePixelRatio=not exposed by Tk, size={self.winfo_screenwidth()}x{self.winfo_screenheight()}"
        )
        for variable in (
            "QT_SCALE_FACTOR", "QT_FONT_DPI", "QT_AUTO_SCREEN_SCALE_FACTOR",
            "QT_ENABLE_HIGHDPI_SCALING",
        ):
            print(f"Environment: {variable}={os.environ.get(variable, '<unset>')}")

        widgets = (
            ("Application default", self, False),
            ("Main window", self, False),
            ("centralWidget", self.central_widget, False),
            ("Input Data title", self.input_frame, True),
            ("Dynamic LQ label", self.dynamic_label, False),
            ("Static UQ label", self.static_label, False),
            ("MRI Modality label", self.modality_label, False),
            ("Path entry", self.dynamic_entry, False),
            ("Browse button", self.dynamic_browse, False),
            ("Advanced Settings", self.advanced_toggle, False),
            ("Workflow title", self.workflow_frame, True),
            ("Workflow Input", self.workflow_name_labels[0], False),
            ("Workflow status", self.workflow_status_labels[0], False),
            ("Preprocessing button", self.preprocess_button, False),
            ("Motion Reconstruction button", self.reconstruct_button, False),
            ("Current Operation title", self.operation_frame, True),
            ("Output Summary title", self.results_frame, True),
            ("Runtime Log title", self.log_frame, True),
            ("Runtime Log text", self.log_text, False),
        )
        for label, widget, title_font in widgets:
            font = self._effective_font(widget, title_font)
            actual = font.actual()
            pixel_size = abs(int(actual["size"])) if int(actual["size"]) < 0 else font.metrics("linespace")
            point_size = pixel_size * 72.0 / logical_dpi
            text_value = ""
            try:
                text_value = str(widget.cget("text"))
            except tk.TclError:
                pass
            print(
                f"{label}: class={widget.winfo_class()}, objectName={widget.winfo_name()}, "
                f"text={text_value!r}, family={actual['family']!r}, requestedSize={actual['size']}, "
                f"effectivePx={pixel_size}, equivalentPt={point_size:.2f}, "
                f"height={widget.winfo_height()}, minimumHeight={widget.winfo_reqheight()}"
            )

    def _on_close(self) -> None:
        if self.controller.busy:
            proceed = messagebox.askyesno(
                "Processing is running",
                "A backend operation is currently running. Closing may interrupt output generation. Continue closing?",
            )
            if not proceed:
                return
        self.destroy()


def main() -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    app = FreeTune4DApp()
    app.mainloop()


if __name__ == "__main__":
    main()
