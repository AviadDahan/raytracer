import numpy as np


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position, dtype=np.float64)
        self.look_at = np.array(look_at, dtype=np.float64)
        self.up_vector = np.array(up_vector, dtype=np.float64)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        
        self.forward = None
        self.right = None
        self.up = None
        self.screen_height = None
    
    def setup(self, aspect_ratio):
        """Compute the orthonormal camera basis and screen dimensions."""
        self.forward = self.look_at - self.position
        self.forward = self.forward / np.linalg.norm(self.forward)
        
        self.right = np.cross(self.forward, self.up_vector)
        self.right = self.right / np.linalg.norm(self.right)
        
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)
        
        self.screen_height = self.screen_width / aspect_ratio
    
    def generate_ray(self, x, y, image_width, image_height):
        """Generate a ray through pixel (x, y)."""
        px = 0.5 - (x + 0.5) / image_width
        py = 0.5 - (y + 0.5) / image_height
        
        px *= self.screen_width
        py *= self.screen_height
        
        screen_center = self.position + self.forward * self.screen_distance
        screen_point = screen_center + self.right * px + self.up * py
        
        direction = screen_point - self.position
        direction = direction / np.linalg.norm(direction)
        
        return self.position, direction
    
    def generate_all_rays(self, image_width, image_height):
        """Generate all rays for the entire image at once (vectorized)."""
        x = np.arange(image_width, dtype=np.float64)
        y = np.arange(image_height, dtype=np.float64)
        
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()
        
        px = 0.5 - (xx + 0.5) / image_width
        py = 0.5 - (yy + 0.5) / image_height
        
        px *= self.screen_width
        py *= self.screen_height
        
        screen_center = self.position + self.forward * self.screen_distance
        screen_points = (screen_center[np.newaxis, :] + 
                        np.outer(px, self.right) + 
                        np.outer(py, self.up))
        
        directions = screen_points - self.position
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        num_rays = image_width * image_height
        origins = np.tile(self.position, (num_rays, 1))
        
        return origins, directions
    
    def generate_rays_for_rows(self, image_width, image_height, y_start, y_end):
        """Generate rays for a range of rows (for parallel rendering)."""
        num_rows = y_end - y_start
        
        x = np.arange(image_width, dtype=np.float64)
        y = np.arange(y_start, y_end, dtype=np.float64)
        
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()
        yy = yy.ravel()
        
        px = 0.5 - (xx + 0.5) / image_width
        py = 0.5 - (yy + 0.5) / image_height
        
        px *= self.screen_width
        py *= self.screen_height
        
        screen_center = self.position + self.forward * self.screen_distance
        screen_points = (screen_center[np.newaxis, :] + 
                        np.outer(px, self.right) + 
                        np.outer(py, self.up))
        
        directions = screen_points - self.position
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        num_rays = num_rows * image_width
        origins = np.tile(self.position, (num_rays, 1))
        
        return origins, directions
