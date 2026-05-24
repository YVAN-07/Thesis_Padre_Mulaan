# side_camera.py
import numpy as np


class SideCamera:
    """
    Thin, Webots-safe wrapper around a Camera device.
    Assumes the Camera is already enabled by the controller.
    Returns RGB images as NumPy arrays.
    """

    def __init__(self, camera):
        """
        Args:
            camera: Webots Camera device (already enabled)
        """
        if camera is None:
            raise RuntimeError("SideCamera received None camera")

        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()

        if self.width == 0 or self.height == 0:
            raise RuntimeError("Camera width/height is zero (not enabled?)")

    def get_image(self):
        """
        Returns:
            np.ndarray (H x W x 3), dtype=uint8
            or None if image not ready
        """
        image = self.camera.getImage()
        if image is None:
            return None

        img = np.frombuffer(image, np.uint8).reshape(
            (self.height, self.width, 4)
        )

        # Drop alpha channel → RGB
        return img[:, :, :3]
