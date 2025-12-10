import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal, dtype=np.float64)
        # Normalize the normal vector
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.offset = offset
        self.material_index = material_index
    
    def intersect(self, ray_origin, ray_direction):
        """
        Compute ray-plane intersection.
        Plane equation: P · N = offset
        
        Returns:
            (t, normal) if intersection found, where t > 0
            (None, None) if no intersection (ray parallel to plane)
        """
        denom = np.dot(ray_direction, self.normal)
        
        # If ray is parallel to plane (or nearly so)
        if abs(denom) < 1e-10:
            return None, None
        
        # t = (offset - origin · normal) / (direction · normal)
        t = (self.offset - np.dot(ray_origin, self.normal)) / denom
        
        if t < 1e-6:
            return None, None
        
        # Normal points towards the ray origin
        if denom > 0:
            normal = -self.normal
        else:
            normal = self.normal
        
        return t, normal
