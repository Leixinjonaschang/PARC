"""Reusable file browser widget for Polyscope's ImGui bindings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import polyscope.imgui as psim


@dataclass
class FileBrowserState:
    """State container for the :func:`file_browser_widget` helper."""

    current_dir: str = field(default_factory=os.getcwd)
    selected_path: Optional[str] = None
    filter_text: str = ""
    show_hidden: bool = False
    root_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self.current_dir = self._normalize_directory(self.current_dir)
        self.set_root_dir(self.root_dir)
        if self.selected_path:
            self.ensure_selection(self.selected_path)

    def _normalize_directory(self, path: Optional[str]) -> str:
        if not path:
            return os.getcwd()
        return os.path.abspath(path)

    def _clamp_to_root(self, path: Optional[str]) -> str:
        candidate = self._normalize_directory(path or (self.root_dir or os.getcwd()))
        if not self.root_dir:
            return candidate
        try:
            common = os.path.commonpath([candidate, self.root_dir])
        except ValueError:
            return self.root_dir
        if common != self.root_dir:
            return self.root_dir
        return candidate

    def set_root_dir(self, root_dir: Optional[str]) -> None:
        if root_dir:
            root_dir = os.path.abspath(root_dir)
            if os.path.isdir(root_dir):
                self.root_dir = root_dir
            else:
                self.root_dir = None
        else:
            self.root_dir = None
        self.current_dir = self._clamp_to_root(self.current_dir)

    def ensure_current_dir(self) -> None:
        self.current_dir = self._clamp_to_root(self.current_dir)
        if os.path.isdir(self.current_dir):
            return
        probe = self.current_dir
        while probe and probe != os.path.dirname(probe):
            probe = os.path.dirname(probe)
            if os.path.isdir(probe):
                self.current_dir = self._clamp_to_root(probe)
                return
        if self.root_dir and os.path.isdir(self.root_dir):
            self.current_dir = self.root_dir
        else:
            self.current_dir = os.getcwd()

    def ensure_selection(self, path: str) -> None:
        if not path:
            return
        abs_path = os.path.abspath(path)
        if self.root_dir:
            try:
                common = os.path.commonpath([abs_path, self.root_dir])
            except ValueError:
                return
            if common != self.root_dir:
                return
        if os.path.isdir(abs_path):
            self.current_dir = self._clamp_to_root(abs_path)
            self.selected_path = abs_path
        elif os.path.isfile(abs_path):
            self.selected_path = abs_path
            parent = os.path.dirname(abs_path) or abs_path
            self.current_dir = self._clamp_to_root(parent)
        else:
            parent = os.path.dirname(abs_path)
            if parent:
                self.current_dir = self._clamp_to_root(parent)

    def refresh(self) -> None:
        self.ensure_current_dir()
        if self.selected_path and not os.path.exists(self.selected_path):
            self.selected_path = None


def _normalize_extensions(extensions: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if not extensions:
        return None
    normalized: List[str] = []
    for ext in extensions:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)
    return tuple(dict.fromkeys(normalized))  # preserve order while removing duplicates


def _gather_entries(
    directory: str,
    show_hidden: bool,
    filter_text: str,
    extensions: Optional[Tuple[str, ...]],
) -> Tuple[List[Tuple[str, str, bool]], Optional[str]]:
    entries: List[Tuple[str, str, bool]] = []
    error: Optional[str] = None
    try:
        with os.scandir(directory) as it:
            for entry in it:
                name = entry.name
                if not show_hidden and name.startswith("."):
                    continue
                is_dir = entry.is_dir()
                if not is_dir and extensions:
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in extensions:
                        continue
                if filter_text and filter_text.lower() not in name.lower():
                    continue
                entries.append((name, entry.path, is_dir))
    except FileNotFoundError:
        error = "Directory not found."
    except PermissionError:
        error = "Permission denied."
    entries.sort(key=lambda item: (not item[2], item[0].lower()))
    return entries, error


def file_browser_widget(
    label: str,
    state: FileBrowserState,
    *,
    root_dir: Optional[str] = None,
    file_extensions: Optional[Sequence[str]] = None,
    select_directories: bool = False,
    child_height: float = 240.0,
) -> bool:
    """Draw a file browser widget.

    Returns ``True`` if a new file or directory was selected during this frame.
    """

    if root_dir is not None:
        state.set_root_dir(root_dir)
    state.refresh()

    normalized_exts = _normalize_extensions(file_extensions)
    selection_changed = False

    up_button_label = f"Up##{label}"
    if psim.Button(up_button_label):
        parent = os.path.dirname(state.current_dir)
        parent = state._clamp_to_root(parent)
        if parent != state.current_dir:
            state.current_dir = parent

    psim.SameLine()
    psim.TextUnformatted(state.current_dir)

    filter_label = f"Filter##{label}"
    changed, new_filter = psim.InputText(filter_label, state.filter_text or "")
    if changed:
        state.filter_text = new_filter

    checkbox_label = f"Show hidden##{label}"
    changed, show_hidden = psim.Checkbox(checkbox_label, state.show_hidden)
    if changed:
        state.show_hidden = show_hidden

    entries, error = _gather_entries(state.current_dir, state.show_hidden, state.filter_text or "", normalized_exts)

    if psim.BeginChild(f"##{label}_entries", [0.0, child_height], border=True):
        if error:
            psim.TextUnformatted(error)
        elif not entries:
            psim.TextUnformatted("No files match the current filters.")
        else:
            for name, path, is_dir in entries:
                display_name = f"[DIR] {name}" if is_dir else name
                is_selected = state.selected_path == path
                clicked, hovered = psim.Selectable(display_name, is_selected)
                double_clicked = hovered and psim.IsMouseDoubleClicked(0)

                if is_dir and (double_clicked or (clicked and not select_directories)):
                    state.current_dir = state._clamp_to_root(path)
                    if select_directories:
                        state.selected_path = path
                        selection_changed = True
                    else:
                        state.selected_path = None
                    continue

                if clicked:
                    if is_dir:
                        state.selected_path = path
                        selection_changed = True
                    else:
                        state.selected_path = path
                        selection_changed = True

                if double_clicked and not is_dir:
                    state.selected_path = path
                    selection_changed = True
        psim.EndChild()

    if state.selected_path:
        psim.TextUnformatted(f"Selected: {state.selected_path}")
    else:
        psim.TextUnformatted("Selected: <none>")

    return selection_changed