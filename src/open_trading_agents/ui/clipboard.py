import os
import shutil
import subprocess
import sys


def copy_desktop_text(text: str) -> bool:
    """Copy text through an installed local desktop clipboard utility.

    Args:
        text: User-selected text to send through standard input.

    Returns:
        Whether a desktop utility accepted the text. Remote sessions use the
        terminal clipboard instead of modifying the server's desktop clipboard.
    """
    if any(os.environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return False

    commands = []
    if sys.platform == "darwin":
        commands.append(("pbcopy",))
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(("wl-copy", "--type", "text/plain;charset=utf-8"))
    if os.environ.get("DISPLAY"):
        commands.extend((("xclip", "-selection", "clipboard", "-in"), ("xsel", "--clipboard", "--input")))

    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            result = subprocess.run(
                (executable, *command[1:]), input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return True
    return False
