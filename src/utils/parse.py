import re
import numpy as np
from collections.abc import Sequence


def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

def isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def is_float(rexpr):
    """Checks if a string matches a float pattern using regex."""
    pattern = r'^[+-]?\d*\.?\d+$'
    return re.match(pattern, rexpr) is not None

def is_nonstring_sequence(obj):
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    return isinstance(obj, Sequence)

def none_or_str(value):
    if value in ('None', 'none', ""):
        return None
    return value


def parse_array(expr: str):
    r"""constructs a numpy array in place of any of the following sub-expressions:
        - x
        - [x1,x2,x3,...]
        - [start : stop (inclusive)] generated via np.arange
        - [start : stop (inclusive) : num] generated via np.linspace
    """
    pattern = r"""^
        ([^\[\]]+)    # capture one or more characters that are NOT [ or ]
        |
        \[
            (
                (?:[^:\[\]]+)        # non-capturing group (?: ) of at least 1 token that is not ':', '[', or ']'
                (?::[^:\[\]]+){0,2}  # non-capturing group of : followed by at least 1 token that is not ':', '[', or ']'
            )
        \]
        $"""
    matches = re.search(pattern, expr, re.VERBOSE)
    match = matches.group(1) or matches.group(2)
    # essentially captures ((?:TOKEN)(?::TOKEN){0,2})
    # ^ ... $           anchors matches to the start and end of string
    # A | B             match A or B
    # [^\[\]]+:         one or more characters that are NOT [ or ]
    # (?:[^:\[\]]+):    non-capturing group (?:) of at least 1 token that is not ':', '[', or ']'
    # (?::[^:\[\]]+):   non-capturing group of : followed by at least 1 token that is not ':', '[', or ']'
    # {0,2}:            repeat previous group 0 to 2 times

    parts = match.split(":")
    if len(parts) == 1:
        xs = parts[0].split(",")
        assert all(isint(x) or isfloat(x) for x in xs)
        if all(isint(x) for x in xs):
            arr = np.array([int(x) for x in xs])
        else:
            arr = np.array([float(x) for x in xs])

    elif len(parts) == 2:
        assert all(isint(x) for x in parts)
        start = int(parts[0])
        stop = int(parts[1])
        arr = np.arange(start, stop+1)

    elif len(parts) == 3:
        assert isfloat(parts[0]) and isfloat(parts[1]) and isint(parts[2])
        start = float(parts[0])
        stop = float(parts[1])
        num = int(parts[2])
        arr = np.linspace(start, stop, num)

    return arr



