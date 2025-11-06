"""
Unicode text transformation utilities.

Provides functions for converting ASCII text to Unicode variants:
superscripts, subscripts, and other typographic transformations.

Useful for:
- Mathematical notation (exponents, indices)
- Chemical formulas (H₂O, CO₂)
- Footnote markers (¹, ², ³)
- Terminal/CLI formatting where markup isn't available
"""


def to_sub(text: str | int | float) -> str:
    """
    Convert text to Unicode subscript characters.

    Useful for chemical formulas, mathematical notation, array indices, etc.

    Args:
        text: String, integer, or float to convert. Non-string types are
              converted via str(). Unsupported characters pass through unchanged.

    Returns:
        String with supported characters converted to subscript Unicode.

    Supported characters:
        - Digits: 0-9 → ₀₁₂₃₄₅₆₇₈₉
        - Operators: + - = ( ) → ₊₋₌₍₎
        - Letters: a, e, o, h, i, j, k, l, m, n, p, r, s, t, u, v, x (limited lowercase only)

    Notes:
        - Unicode has very limited subscript letter support
        - Unsupported characters pass through unchanged
        - Missing punctuation: . , : ; ! ? ' " '/' '\' @ # $ % ^ & * _ ~ ` [ ] { } < > | and space
        - Missing letters: uppercase A–Z lowercase b, c, d, f, g, q, w, y, z

    Examples:
        >>> to_sub(2)
        '₂'

        >>> to_sub("H2O")
        'H₂O'

        >>> to_sub("x(n+1)")
        'ₓ₍ₙ₊₁₎'

        >>> to_sub("CO2")
        'CO₂'

    See Also:
        to_sup() - Companion function for superscript conversion
    """
    text = str(text)

    subscript_map = {
        # Digits
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        # Operators and punctuation
        "+": "₊",
        "-": "₋",
        "=": "₌",
        "(": "₍",
        ")": "₎",
        # Limited letter support in Unicode
        "a": "ₐ",
        "e": "ₑ",
        "o": "ₒ",
        "h": "ₕ",
        "i": "ᵢ",
        "j": "ⱼ",
        "k": "ₖ",
        "l": "ₗ",
        "m": "ₘ",
        "n": "ₙ",
        "p": "ₚ",
        "r": "ᵣ",
        "s": "ₛ",
        "t": "ₜ",
        "u": "ᵤ",
        "v": "ᵥ",
        "x": "ₓ",
    }

    return "".join(subscript_map.get(c, c) for c in text)


def to_sup(text: str | int | float) -> str:
    """
    Convert text to Unicode superscript characters.

    Supports digits, common operators, parentheses, and letters.
    Useful for mathematical notation, footnotes, exponents, etc.

    Args:
        text: String, integer, or float to convert. Non-string types are
              converted via str(). Unsupported characters pass through unchanged.

    Returns:
        String with supported characters converted to superscript Unicode.

    Supported characters:
        - Digits: 0-9 → ⁰¹²³⁴⁵⁶⁷⁸⁹
        - Operators: + - = ( ) → ⁺⁻⁼⁽⁾
        - Letters: a-z, A-Z (partial support, see notes)
        - Space → (regular space, no superscript variant)

    Unsupported characters:
        - Unsupported characters are left unchanged
        - Missing 'q' (left unchanged)
        - Missing punctuations (left unchanged): . , : ; ! ? ' " '/' '\' @ # $ % ^ & * _ ~ ` [ ] { } < > | and space
        - Missing uppercase letters (left unchanged): C, F, Q, S, X, Y, Z

    Examples:
        >>> to_sup(3)
        '³'

        >>> to_sup(-12)
        '⁻¹²'

        >>> to_sup("(n+1)")
        '⁽ⁿ⁺¹⁾'

        >>> to_sup("x2")
        'ˣ²'

        >>> to_sup("QWERTY")
        'QᵂᴱᴿᵀY'

    See Also:
        to_sub(): - Convert to Unicode subscript characters
    """
    text = str(text)

    superscript_map = {
        # Digits
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        # Operators and punctuation
        "+": "⁺",
        "-": "⁻",
        "=": "⁼",
        "(": "⁽",
        ")": "⁾",
        # Lowercase letters (limited Unicode support)
        "a": "ᵃ",
        "b": "ᵇ",
        "c": "ᶜ",
        "d": "ᵈ",
        "e": "ᵉ",
        "f": "ᶠ",
        "g": "ᵍ",
        "h": "ʰ",
        "i": "ⁱ",
        "j": "ʲ",
        "k": "ᵏ",
        "l": "ˡ",
        "m": "ᵐ",
        "n": "ⁿ",
        "o": "ᵒ",
        "p": "ᵖ",
        "r": "ʳ",
        "s": "ˢ",
        "t": "ᵗ",
        "u": "ᵘ",
        "v": "ᵛ",
        "w": "ʷ",
        "x": "ˣ",
        "y": "ʸ",
        "z": "ᶻ",
        # Note: q has no standardsuperscript
        # Uppercase letters (limited Unicode support)
        "A": "ᴬ",
        "B": "ᴮ",
        "D": "ᴰ",
        "E": "ᴱ",
        "G": "ᴳ",
        "H": "ᴴ",
        "I": "ᴵ",
        "J": "ᴶ",
        "K": "ᴷ",
        "L": "ᴸ",
        "M": "ᴹ",
        "N": "ᴺ",
        "O": "ᴼ",
        "P": "ᴾ",
        "R": "ᴿ",
        "T": "ᵀ",
        "U": "ᵁ",
        "V": "ⱽ",
        "W": "ᵂ",
        # Note: C, F, Q, S, X, Y, Z have no standardsuperscripts
    }

    return "".join(superscript_map.get(c, c) for c in text)
