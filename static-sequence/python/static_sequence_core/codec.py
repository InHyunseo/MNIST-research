"""Conversion between three digit tokens and display strings."""

from collections.abc import Iterable

SEQUENCE_LENGTH = 3


def _validated_digits(digits: Iterable[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in digits)
    if len(values) != SEQUENCE_LENGTH:
        raise ValueError(f"expected {SEQUENCE_LENGTH} digits, got {len(values)}")
    if any(value < 0 or value > 9 for value in values):
        raise ValueError(f"digits must be in [0, 9], got {values}")
    return values  # type: ignore[return-value]


def digits_to_string(digits: Iterable[int]) -> str:
    """Format exactly three digits without losing leading zeros."""
    return "".join(str(value) for value in _validated_digits(digits))


def string_to_digits(text: str) -> tuple[int, int, int]:
    if len(text) != SEQUENCE_LENGTH or not text.isascii() or not text.isdigit():
        raise ValueError(f"expected a three-character ASCII digit string, got {text!r}")
    return _validated_digits(int(character) for character in text)
