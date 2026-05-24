# webots_progress_overly.py
def draw_bar(display, y, value, label, width=300):
    """
    Draws a horizontal progress bar on Webots Display.
    value expected in [-1, 1] or [0, 1].
    """

    # Clamp
    value = max(-1.0, min(1.0, value))

    # Background
    display.setColor(0x444444)
    display.fillRectangle(20, y, width, 15)

    # Foreground
    bar_width = int((value + 1) / 2 * width)
    color = 0x00FF00 if value >= 0 else 0xFF0000
    display.setColor(color)
    display.fillRectangle(20, y, bar_width, 15)

    # Label
    display.setColor(0xFFFFFF)
    display.drawText(f"{label}: {value:+.3f}", 20, y - 5)
