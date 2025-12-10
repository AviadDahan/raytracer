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
    
    def generate_all_rays(self, image_width, image_height):
        """
        Generate all rays for the entire image at once (vectorized).
        
        Returns:
            ray_origins: (H*W, 3) array of ray origins (all same, but tiled for convenience)
            ray_directions: (H*W, 3) array of normalized ray directions
        """
        # Create pixel coordinate grids
        x = np.arange(image_width, dtype=np.float64)
        y = np.arange(image_height, dtype=np.float64)
        
        # Create meshgrid: xx[i,j] = j (column), yy[i,j] = i (row)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten to 1D arrays of shape (H*W,)
        xx = xx.ravel()
        yy = yy.ravel()
        
        # Convert pixel coordinates to normalized screen coordinates [-0.5, 0.5]
        px = (xx + 0.5) / image_width - 0.5
        py = 0.5 - (yy + 0.5) / image_height
        
        # Scale to screen dimensions
        px *= self.screen_width
        py *= self.screen_height
        
        # Compute screen points: (H*W, 3)
        # screen_center is (3,), we need to broadcast
        screen_center = self.position + self.forward * self.screen_distance
        
        # Build direction vectors using outer products
        # screen_point = screen_center + right * px + up * py
        # px and py are (H*W,), right and up are (3,)
        screen_points = (screen_center[np.newaxis, :] + 
                        np.outer(px, self.right) + 
                        np.outer(py, self.up))
        
        # Ray directions (H*W, 3)
        directions = screen_points - self.position
        
        # Normalize directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        # Ray origins: all the same, tile for convenience
        num_rays = image_width * image_height
        origins = np.tile(self.position, (num_rays, 1))
        
        return origins, directions
    
    def generate_rays_for_rows(self, image_width, image_height, y_start, y_end):
        """
        Generate rays for a specific range of rows (for parallel rendering).
        
        Args:
            image_width: full image width
            image_height: full image height
            y_start: starting row (inclusive)
            y_end: ending row (exclusive)
        
        Returns:
            ray_origins: (num_rows * width, 3) array of ray origins
            ray_directions: (num_rows * width, 3) array of normalized ray directions
        """
        num_rows = y_end - y_start
        
        # Create pixel coordinate grids for the specified rows
        x = np.arange(image_width, dtype=np.float64)
        y = np.arange(y_start, y_end, dtype=np.float64)
        
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()
        
        # Convert to normalized screen coordinates
        px = (xx + 0.5) / image_width - 0.5
        py = 0.5 - (yy + 0.5) / image_height
        
        px *= self.screen_width
        py *= self.screen_height
        
        # Compute screen points
        screen_center = self.position + self.forward * self.screen_distance
        screen_points = (screen_center[np.newaxis, :] + 
                        np.outer(px, self.right) + 
                        np.outer(py, self.up))
        
        # Ray directions
        directions = screen_points - self.position
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        # Ray origins
        num_rays = num_rows * image_width
        origins = np.tile(self.position, (num_rays, 1))
        
        return origins, directions