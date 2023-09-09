
def pair(t):
    """Wraps t as tuple if not already tuple."""
    return t if isinstance(t, tuple) else (t, t)
