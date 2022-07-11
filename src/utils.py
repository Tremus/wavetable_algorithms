def skew(x: float, a: float) -> float:
    if a >= 1: return 1
    if a <= 0: return 0
    b = a * x;
    return b / (1 - a - x + b + b)
