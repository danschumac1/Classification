import numpy as np

def letcs_transform(ts_list: list, precision: int = 3) -> str:
    """
    Convert a time series list (list of floats or ints) into LETCS-style
    digit-space formatting.
    """
    formatted_steps = []

    for x in ts_list:
        s = f"{float(x):.{precision}f}"
        s = s.replace(".", "")
        s = s.replace(",", "")
        digit_spaced = " ".join(list(s))
        formatted_steps.append(digit_spaced)

    return " , ".join(formatted_steps)


def letcs_transform_multivar(ts_2d, precision: int = 3) -> str:
    """
    Convert a multivariate time series (any ndim >= 1) into a single
    LETCS-style string by flattening all dimensions.
    """
    arr = np.asarray(ts_2d, dtype=float)   # ensures numeric + handles nested lists
    flat = arr.flatten().tolist()          # 1D list of floats
    return letcs_transform(flat, precision=precision)
