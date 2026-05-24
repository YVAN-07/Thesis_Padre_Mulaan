# ema.py
class EMA:
    """
    Exponential Moving Average.
    Stores exactly ONE value.
    """

    def __init__(self, alpha: float):
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha
        self.prev = None

    def update(self, value: float) -> float:
        if self.prev is None:
            self.prev = value
        else:
            self.prev = self.alpha * value + (1.0 - self.alpha) * self.prev
        return self.prev
