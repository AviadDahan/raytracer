import numpy as np


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = np.array(position, dtype=np.float64)
        self.radius = radius
        self.material_index = material_index
    
    def intersect(self, ray_origin, ray_direction):
        """
        Compute ray-sphere intersection using the quadratic formula.
        
        Returns:
            (t, normal) if intersection found, where t > 0
            (None, None) if no intersection
        """
        # Vector from ray origin to sphere center
        oc = ray_origin - self.position
        
        # Quadratic coefficients: at² + bt + c = 0
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None, None
        
        sqrt_disc = np.sqrt(discriminant)
        
        # Find the nearest positive t
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        # We want the smallest positive t
        if t1 > 1e-6:
            t = t1
        elif t2 > 1e-6:
            t = t2
        else:
            return None, None
        
        # Compute hit point and normal
        hit_point = ray_origin + t * ray_direction
        normal = (hit_point - self.position) / self.radius
        
        return t, normal
