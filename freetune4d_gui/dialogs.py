"""Application-controlled, readable file-system dialogs."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

from .typography import TYPOGRAPHY


class DirectoryDialog(tk.Toplevel):
    """A non-native directory chooser that inherits FreeTune4D typography."""

    def __init__(self, parent: tk.Misc, initial_directory: Path):
        super().__init__(parent)
        self.result: Path | None = None
        self.current = self._existing_directory(initial_directory)
        self.title("Select Directory")
        self.geometry("850x600")
        self.minsize(650, 450)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        ttk.Label(body, text="Current folder", style="StatusName.TLabel").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(body, textvariable=self.path_var)
        self.path_entry.grid(row=1, column=0, sticky="ew", pady=(5, 10))
        self.path_entry.bind("<Return>", self._go_to_typed_path)

        self.tree = ttk.Treeview(body, show="tree", selectmode="browse", style="Directory.Treeview")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=2, column=0, sticky="nsew")
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.tree.bind("<Double-1>", self._open_selected)
        self.tree.bind("<Return>", self._open_selected)

        actions = ttk.Frame(body)
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(actions, text="Up", command=self._go_up).pack(side="left")
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(actions, text="Select Folder", command=self._accept).pack(side="right", padx=(0, 10))

        style = ttk.Style(self)
        style.configure(
            "Directory.Treeview", font="FreeTune4DMedium",
            rowheight=max(TYPOGRAPHY.CONTROL_HEIGHT_PX, TYPOGRAPHY.BODY_PX + 14),
        )
        self._populate()
        self.grab_set()
        self.path_entry.focus_set()

    @staticmethod
    def _existing_directory(path: Path) -> Path:
        candidate = path.expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        while not candidate.is_dir() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate.resolve() if candidate.is_dir() else Path.home().resolve()

    def _populate(self) -> None:
        self.path_var.set(str(self.current))
        self.tree.delete(*self.tree.get_children())
        try:
            directories = sorted(
                (path for path in self.current.iterdir() if path.is_dir()),
                key=lambda path: path.name.casefold(),
            )
        except OSError:
            directories = []
        for directory in directories:
            self.tree.insert("", "end", iid=str(directory), text=directory.name)

    def _selected_path(self) -> Path | None:
        selection = self.tree.selection()
        return Path(selection[0]) if selection else None

    def _open_selected(self, _event=None) -> None:
        selected = self._selected_path()
        if selected and selected.is_dir():
            self.current = selected.resolve()
            self._populate()

    def _go_up(self) -> None:
        self.current = self.current.parent
        self._populate()

    def _go_to_typed_path(self, _event=None) -> None:
        typed = Path(self.path_var.get()).expanduser()
        if typed.is_dir():
            self.current = typed.resolve()
            self._populate()

    def _accept(self) -> None:
        selected = self._selected_path()
        self.result = (selected or self.current).resolve()
        self.destroy()


def choose_directory(parent: tk.Misc, initial_directory: Path) -> Path | None:
    dialog = DirectoryDialog(parent, initial_directory)
    parent.wait_window(dialog)
    return dialog.result
