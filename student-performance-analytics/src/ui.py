"""Small terminal "theme" helpers: colours, banners, tables, safe input.

Nothing here is related to data analysis - it only makes the command-line
application pleasant to read. Colours are switched off automatically when the
output is not a real terminal, or when the NO_COLOR environment variable is set.
"""

import os
import sys
import textwrap

# Make Unicode (box drawing, arrows) safe on Windows consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Enables ANSI colour codes in the classic Windows console as well.
if os.name == "nt":
    os.system("")

USE_COLOUR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
WIDTH = 78

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[38;5;37m", "green": "\033[38;5;71m",
    "yellow": "\033[38;5;179m", "red": "\033[38;5;167m", "grey": "\033[38;5;245m",
}


def paint(text, *styles):
    """Wrap text in ANSI colour codes (or return it unchanged)."""
    if not USE_COLOUR:
        return str(text)
    start = "".join(_CODES[s] for s in styles)
    return f"{start}{text}{_CODES['reset']}"


def banner():
    lines = [
        "STUDENT PERFORMANCE & DATA ANALYTICS SYSTEM",
        "Python  |  NumPy  |  Pandas  |  Matplotlib",
    ]
    print()
    print(paint("╔" + "═" * (WIDTH - 2) + "╗", "cyan"))
    for i, line in enumerate(lines):
        style = ("bold", "cyan") if i == 0 else ("grey",)
        print(paint("║", "cyan") + paint(line.center(WIDTH - 2), *style) + paint("║", "cyan"))
    print(paint("╚" + "═" * (WIDTH - 2) + "╝", "cyan"))


def header(title):
    print()
    print(paint(f"▌ {title}", "bold", "cyan"))
    print(paint("─" * WIDTH, "grey"))


def subheader(title):
    print()
    print(paint(f"  {title}", "bold"))


def _wrap(message, indent="    ", hang=0):
    return textwrap.fill(str(message), width=WIDTH, initial_indent="",
                         subsequent_indent=indent + " " * hang)


def info(message):
    print(paint("  ℹ ", "cyan") + _wrap(message, hang=0))


def success(message):
    print(paint("  ✔ ", "green") + _wrap(message))


def warning(message):
    print(paint("  ⚠ ", "yellow") + _wrap(message))


def error(message):
    print(paint("  ✖ ", "red") + _wrap(message))


def note(message, hang=0):
    text = textwrap.fill(str(message), width=WIDTH, initial_indent="    ",
                         subsequent_indent="    " + " " * hang)
    print(paint(text, "dim"))


def show_table(df, max_rows=None, float_digits=2):
    """Print a DataFrame with tidy number formatting and an indent."""
    if df is None or len(df) == 0:
        note("(no rows to display)")
        return
    text = df.to_string(
        index=False,
        max_rows=max_rows,
        float_format=lambda x: f"{x:.{float_digits}f}",
    )
    for line in text.splitlines():
        print("  " + line)


def show_key_values(pairs, key_width=30):
    value_width = max(20, WIDTH - key_width - 3)
    for key, value in pairs:
        lines = textwrap.wrap(str(value), width=value_width) or [""]
        print(f"  {paint(str(key).ljust(key_width), 'grey')} {lines[0]}")
        for extra in lines[1:]:
            print(" " * (key_width + 3) + extra)


def menu(title, options):
    """Print a numbered menu. `options` is a list of (key, label)."""
    header(title)
    for key, label in options:
        print(f"  {paint(str(key).rjust(2), 'bold', 'yellow')}  {label}")
    print()


def ask(prompt, valid=None, default=None):
    """Read input safely. Re-asks until the answer is in `valid` (if given)."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            answer = input(paint(f"  ➜ {prompt}{suffix}: ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "exit"
        if not answer and default is not None:
            return str(default)
        if valid is None or answer.lower() in [str(v).lower() for v in valid]:
            return answer
        error(f"Please enter one of: {', '.join(str(v) for v in valid)}")


def ask_int(prompt, minimum, maximum, default=None):
    """Ask for a whole number inside [minimum, maximum]."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            raw = input(paint(f"  ➜ {prompt} ({minimum}-{maximum}){suffix}: ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default if default is not None else minimum
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            error("That is not a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        error(f"Number must be between {minimum} and {maximum}.")


def pause():
    try:
        input(paint("\n  Press Enter to continue...", "grey"))
    except (EOFError, KeyboardInterrupt):
        print()


def ask_float(prompt, minimum, maximum, default=None):
    """Ask for a number inside [minimum, maximum]."""
    suffix = f" [{default:g}]" if default is not None else ""
    while True:
        try:
            raw = input(paint(f"  ➜ {prompt} ({minimum:g}-{maximum:g}){suffix}: ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default if default is not None else minimum
        if not raw and default is not None:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            error("That is not a number.")
            continue
        if minimum <= value <= maximum:
            return value
        error(f"Number must be between {minimum:g} and {maximum:g}.")
