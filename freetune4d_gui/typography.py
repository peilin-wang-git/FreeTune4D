"""Single, pixel-verified typography system for the Tk GUI."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
import tkinter.font as tkfont


@dataclass(frozen=True)
class Typography:
    """Visual font and control targets, expressed in screen pixels."""

    BODY_PX: int = 17
    SECTION_PX: int = 20
    TITLE_PX: int = 24
    LOG_PX: int = 15
    CONTROL_HEIGHT_PX: int = 38
    PRIMARY_HEIGHT_PX: int = 46


TYPOGRAPHY = Typography()


def configure_named_fonts(root: tk.Misc) -> dict[str, tkfont.Font]:
    """Configure the only fonts used by the application.

    Tk interprets a negative font size as pixels.  This is intentional: the
    workstation screenshots demonstrated that point sizes mediated by Tk's
    ``tk scaling`` value did not reach the requested visual size.
    """

    base = tkfont.nametofont("TkDefaultFont", root=root)
    family = base.actual("family")
    fixed_family = tkfont.nametofont("TkFixedFont", root=root).actual("family")

    roles = {
        "body": tkfont.Font(root=root, name="FreeTune4DBody", family=family, size=-TYPOGRAPHY.BODY_PX),
        "body_bold": tkfont.Font(root=root, name="FreeTune4DBodyBold", family=family, size=-TYPOGRAPHY.BODY_PX, weight="bold"),
        "section": tkfont.Font(root=root, name="FreeTune4DSection", family=family, size=-TYPOGRAPHY.SECTION_PX, weight="bold"),
        "title": tkfont.Font(root=root, name="FreeTune4DTitle", family=family, size=-TYPOGRAPHY.TITLE_PX, weight="bold"),
        "log": tkfont.Font(root=root, name="FreeTune4DLog", family=fixed_family, size=-TYPOGRAPHY.LOG_PX),
    }

    # Classic Tk widgets and dialogs inherit the same body role.  ttk widgets
    # receive these exact named roles from the central style setup.
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkCaptionFont", "TkSmallCaptionFont"):
        tkfont.nametofont(name, root=root).configure(family=family, size=-TYPOGRAPHY.BODY_PX)
    tkfont.nametofont("TkHeadingFont", root=root).configure(
        family=family, size=-TYPOGRAPHY.SECTION_PX, weight="bold"
    )
    tkfont.nametofont("TkFixedFont", root=root).configure(family=fixed_family, size=-TYPOGRAPHY.LOG_PX)
    return roles
