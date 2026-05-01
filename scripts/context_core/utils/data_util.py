"""Minimal file utilities for the local evaluation pipeline."""


def save_to_file(text: str, filename: str = "error_output.txt") -> None:
    """Append a single line of text to a file."""
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{text}\n")
