import numpy as np


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position, dtype=np.float64)
        self.look_at = np.array(look_at, dtype=np.float64)
        self.up_vector = np.array(up_vector, dtype=np.float64)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        
        # These will be computed in setup()
        self.forward = None
        self.right = None
        self.up = None
        self.screen_height = None
    
    def setup(self, aspect_ratio):
        """
        Compute the orthonormal basis for the camera and screen dimensions.
        aspect_ratio = width / height
        """
        # Forward vector: from camera position to look-at point
        self.forward = self.look_at - self.position
        self.forward = self.forward / np.linalg.norm(self.forward)
        
        # Right vector: perpendicular to forward and up
        self.right = np.cross(self.forward, self.up_vector)
        self.right = self.right / np.linalg.norm(self.right)
        
        # Up vector: perpendicular to forward and right (fixes the up vector)
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)
        
        # Compute screen height from aspect ratio
        self.screen_height = self.screen_width / aspect_ratio
    
    def generate_ray(self, x, y, image_width, image_height):
        """
        Generate a ray through pixel (x, y).
        x, y are pixel coordinates (0-indexed from top-left).
        Returns ray origin and normalized direction.
        """
        # Convert pixel coordinates to normalized screen coordinates [-0.5, 0.5]
        # Center of pixel
        px = (x + 0.5) / image_width - 0.5   # ranges from -0.5 to 0.5
        py = 0.5 - (y + 0.5) / image_height  # flip y, ranges from -0.5 to 0.5
        
        # Scale to screen dimensions
        px *= self.screen_width
        py *= self.screen_height
        
        # Compute the point on the screen
        screen_center = self.position + self.forward * self.screen_distance
        screen_point = screen_center + self.right * px + self.up * py
        
        # Ray direction
        direction = screen_point - self.position
        direction = direction / np.linalg.norm(direction)
        
        return self.position, direction
