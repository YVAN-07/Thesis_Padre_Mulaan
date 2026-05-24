# window_buffer.py
class WindowBuffer:
    """
    Fixed-size rolling buffer for normalization.
    Tracks min/max of recent values with robust zero-handling.
    
    Used to normalize bidirectional deltas (very small changes in similarity).
    Properly handles near-zero ranges when changes are minimal.
    """

    def __init__(self, size: int):
        """
        Args:
            size: Maximum number of values to keep (default 10 frames)
        """
        self.size = size
        self.buffer = []

    def add(self, value: float):
        """
        Add a value to the buffer (e.g., delta in similarity).
        Maintains rolling window of most recent values.
        
        Args:
            value: The delta/change value to store
        """
        self.buffer.append(value)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def min(self):
        """
        Get minimum value in buffer.
        Returns 0.0 if buffer is empty.
        """
        return min(self.buffer) if self.buffer else 0.0

    def max(self):
        """
        Get maximum value in buffer.
        Returns 1.0 if buffer is empty (safe default).
        """
        return max(self.buffer) if self.buffer else 1.0
    
    def range(self):
        """
        Get the span (max - min) of values in buffer.
        Returns max - min if buffer has data, else returns -1 for empty.
        """
        if len(self.buffer) == 0:
            return -1.0
        return self.max() - self.min()
    
    def is_empty(self):
        """
        Check if buffer is empty.
        """
        return len(self.buffer) == 0
    
    def count(self):
        """
        Get current number of values stored.
        """
        return len(self.buffer)
