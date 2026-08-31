"""Tkinter desktop application for orchestrating FreeTune4D."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .backend import BackendError, FreeTune4DBackend, OutputSummary, PipelineConfig
from .controller import WorkflowController, WorkflowState


class FreeTune4DApp(tk.Tk):
    POLL_MS = 100

    def __init__(self, backend: FreeTune4DBackend | None = None):
        super().__init__()
        self.backend = backend or FreeTune4DBackend()
        self.controller = WorkflowController()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_validation_errors: list[str] = []
        self.output_summary: OutputSummary | None = None

        self.title("FreeTune4D — UQ 4D-MRI Motion Reconstruction")
        self.geometry("980x800")
        self.minsize(860, 680)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._create_variables()
        self._configure_style()
        self._build_ui()
        self.after(self.POLL_MS, self._poll_events)
        self._validate_form(log_success=False)

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
        ):
            variable.trace_add("write", self._on_config_changed)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Header.TLabel", font=("TkDefaultFont", 20, "bold"), foreground="#17324d")
        style.configure("Subheader.TLabel", font=("TkDefaultFont", 10), foreground="#526577")
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 11, "bold"), foreground="#17324d")
        style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"), padding=(14, 7))
        style.configure("Status.TLabel", padding=(8, 4), background="#edf2f6")
        style.configure("Success.TLabel", foreground="#176b3a")
        style.configure("Error.TLabel", foreground="#a32929")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="FreeTune4D", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="UQ 4D-MRI Motion Reconstruction", style="Subheader.TLabel").pack(anchor="w")

        panes = ttk.Panedwindow(outer, orient="vertical")
        panes.pack(fill="both", expand=True)
        main = ttk.Frame(panes)
        log_frame = ttk.Labelframe(panes, text="Runtime Log", style="Section.TLabelframe")
        panes.add(main, weight=4)
        panes.add(log_frame, weight=1)

        input_frame = ttk.Labelframe(main, text="Input Data", padding=10, style="Section.TLabelframe")
        input_frame.pack(fill="x", pady=(0, 8))
        self.dynamic_entry, self.dynamic_browse = self._path_row(
            input_frame, 0, "Dynamic LQ 4D-MRI DICOM",
            "Respiratory-resolved low-quality 4D-MRI containing motion information.",
            self.dynamic_var, lambda: self._browse_directory(self.dynamic_var),
        )
        self.static_entry, self.static_browse = self._path_row(
            input_frame, 2, "Static UQ 3D-MRI DICOM",
            "High-quality 3D MRI providing anatomical prior information.",
            self.static_var, lambda: self._browse_directory(self.static_var),
        )
        ttk.Label(input_frame, text="MRI Modality").grid(row=4, column=0, sticky="w", pady=(8, 2))
        self.modality_combo = ttk.Combobox(input_frame, textvariable=self.modality_var, values=("T2-weighted", "T1-weighted"), state="readonly", width=24)
        self.modality_combo.grid(row=5, column=0, sticky="w")
        self.modality_note = ttk.Label(input_frame, text="T2 backend available. T1 backend is not present in this repository.", foreground="#7a5a00")
        self.modality_note.grid(row=5, column=1, sticky="w", padx=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=0)

        output_frame = ttk.Labelframe(main, text="Output", padding=10, style="Section.TLabelframe")
        output_frame.pack(fill="x", pady=(0, 8))
        self.output_entry, self.output_browse = self._path_row(
            output_frame, 0, "Output Directory", "The application creates preprocessing, reconstructed and QC below this root.",
            self.output_var, lambda: self._browse_directory(self.output_var),
        )
        output_frame.columnconfigure(0, weight=1)

        self.advanced = ttk.Labelframe(main, text="Advanced Settings", padding=10, style="Section.TLabelframe")
        self.advanced.pack(fill="x", pady=(0, 8))
        self.advanced_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.advanced, text="Show model and backend settings", variable=self.advanced_visible, command=self._toggle_advanced).grid(row=0, column=0, sticky="w")
        self.advanced_body = ttk.Frame(self.advanced)
        self._file_row(self.advanced_body, 0, "Coarse model (coarse.h5)", self.coarse_var, lambda: self._browse_file(self.coarse_var, (("H5 model", "*.h5"),)))
        self._file_row(self.advanced_body, 1, "Fine model (fine.h5)", self.fine_var, lambda: self._browse_file(self.fine_var, (("H5 model", "*.h5"),)))
        self._file_row(self.advanced_body, 2, "Reference DICOM (optional)", self.reference_var, lambda: self._browse_file(self.reference_var, (("DICOM", "*.dcm"), ("All files", "*"))))
        ttk.Label(self.advanced_body, text="Respiratory phases").grid(row=3, column=0, sticky="w", pady=4)
        self.phase_spin = ttk.Spinbox(self.advanced_body, from_=1, to=65, textvariable=self.phase_var, width=8)
        self.phase_spin.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=4)
        self.advanced_body.columnconfigure(1, weight=1)

        workflow = ttk.Labelframe(main, text="Reconstruction Workflow", padding=10, style="Section.TLabelframe")
        workflow.pack(fill="x", pady=(0, 8))
        status_grid = ttk.Frame(workflow)
        status_grid.pack(fill="x")
        for row, (number, title, variable) in enumerate((
            ("1", "Input", self.input_status_var),
            ("2", "Preprocessing", self.preprocess_status_var),
            ("3", "Motion Reconstruction", self.reconstruct_status_var),
            ("4", "Output", self.output_status_var),
        )):
            ttk.Label(status_grid, text=number, width=3).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(status_grid, text=title, width=25).grid(row=row, column=1, sticky="w", pady=2)
            ttk.Label(status_grid, textvariable=variable, style="Status.TLabel", width=18).grid(row=row, column=2, sticky="w", pady=2)
        buttons = ttk.Frame(workflow)
        buttons.pack(fill="x", pady=(10, 0))
        self.preprocess_button = ttk.Button(buttons, text="Preprocessing", style="Primary.TButton", command=self._start_preprocessing)
        self.preprocess_button.pack(side="left")
        self.reconstruct_button = ttk.Button(buttons, text="Motion Reconstruction", style="Primary.TButton", command=self._start_reconstruction)
        self.reconstruct_button.pack(side="left", padx=(10, 0))
        self.activity = ttk.Progressbar(buttons, mode="indeterminate", length=170)
        self.activity.pack(side="right")
        ttk.Label(workflow, textvariable=self.overall_status_var).pack(anchor="w", pady=(8, 0))

        results = ttk.Labelframe(main, text="Results / Output Summary", padding=10, style="Section.TLabelframe")
        results.pack(fill="x")
        self.results_text = tk.Text(results, height=4, wrap="word", state="disabled", background="#f7f9fb", relief="flat")
        self.results_text.pack(fill="x")
        actions = ttk.Frame(results)
        actions.pack(fill="x", pady=(6, 0))
        self.open_output_button = ttk.Button(actions, text="Open Output Folder", command=lambda: self._open_folder(self._config().output_root))
        self.open_reconstructed_button = ttk.Button(actions, text="Open Reconstructed Folder", command=lambda: self._open_folder(self.backend.output_paths(self._config())[1]))
        self.open_qc_button = ttk.Button(actions, text="Open QC Folder", command=lambda: self._open_folder(self.backend.output_paths(self._config())[2]))
        for button in (self.open_output_button, self.open_reconstructed_button, self.open_qc_button):
            button.pack(side="left", padx=(0, 8))

        self.log_text = tk.Text(log_frame, height=8, wrap="none", state="disabled", background="#101820", foreground="#dce5ec", insertbackground="white")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        self._update_controls()

    def _path_row(self, parent, row, label, description, variable, browse_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Label(parent, text=description, foreground="#657786").grid(row=row + 1, column=0, sticky="w", pady=(0, 4))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row + 1, column=0, sticky="ew", padx=(0, 8), pady=(18, 4))
        button = ttk.Button(parent, text="Browse...", command=browse_command)
        button.grid(row=row + 1, column=1, sticky="e", pady=(18, 4))
        return entry, button

    def _file_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        button = ttk.Button(parent, text="Browse...", command=command)
        button.grid(row=row, column=2, pady=4)
        if not hasattr(self, "advanced_controls"):
            self.advanced_controls = []
        self.advanced_controls.extend((entry, button))

    def _toggle_advanced(self) -> None:
        if self.advanced_visible.get():
            self.advanced_body.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            self.advanced.columnconfigure(0, weight=1)
        else:
            self.advanced_body.grid_remove()

    def _browse_directory(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(Path.home()))
        if selected:
            variable.set(selected)

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
        )

    def _on_config_changed(self, *_args) -> None:
        if hasattr(self, "preprocess_button") and not self.controller.busy:
            self._validate_form(log_success=False)

    def _validate_form(self, log_success: bool = True) -> bool:
        config = self._config()
        errors = self.backend.validate_inputs(config)
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
                    self._log(str(payload))
                elif event == "success":
                    self._operation_succeeded(*payload)
                elif event == "failure":
                    self._operation_failed(*payload)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._poll_events)

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
        self.overall_status_var.set(f"{label} failed: {exc}")
        self._log(f"ERROR: {label} failed: {exc}")
        self._log(details)
        messagebox.showerror("Operation failed", f"{label} failed.\n\n{exc}\n\nDetailed traceback is available in Runtime Log.")
        self.controller.recover_after_failure()
        self._update_controls()

    def _update_controls(self) -> None:
        busy = self.controller.busy
        editable_state = "disabled" if busy else "normal"
        for entry in (self.dynamic_entry, self.static_entry, self.output_entry):
            entry.configure(state=editable_state)
        for button in (self.dynamic_browse, self.static_browse, self.output_browse):
            button.configure(state=editable_state)
        self.modality_combo.configure(state="disabled" if busy else "readonly")
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
        self.reconstruct_status_var.set({WorkflowState.RECONSTRUCTING: "Running...", WorkflowState.COMPLETED: "Completed"}.get(state, "Waiting"))
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

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
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
    app = FreeTune4DApp()
    app.mainloop()


if __name__ == "__main__":
    main()
