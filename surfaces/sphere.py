import numpy as np


class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = np.array(position, dtype=np.float64)
        self.radius = radius
        self.material_index = material_index
    
    def intersect(self, ray_origin, ray_direction):
        """Compute ray-sphere intersection using the quadratic formula."""
        oc = ray_origin - self.position
        
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None, None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        if t1 > 1e-6:
            t = t1
        elif t2 > 1e-6:
            t = t2
        else:
            return None, None
        
        hit_point = ray_origin + t * ray_direction
        normal = (hit_point - self.position) / self.radius
        
        return t, normal
    
    def intersect_batch(self, ray_origins, ray_directions):
        """Vectorized ray-sphere intersection for N rays."""
        N = ray_origins.shape[0]
        
        oc = ray_origins - self.position
        a = np.sum(ray_directions * ray_directions, axis=1)
        b = 2.0 * np.sum(oc * ray_directions, axis=1)
        c = np.sum(oc * oc, axis=1) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        
        t_values = np.full(N, np.inf)
        normals = np.zeros((N, 3))
        
        hit_mask = discriminant >= 0
        if not np.any(hit_mask):
            return t_values, normals
        
        sqrt_disc = np.sqrt(discriminant[hit_mask])
        a_hit = a[hit_mask]
        b_hit = b[hit_mask]
        
        t1 = (-b_hit - sqrt_disc) / (2 * a_hit)
        t2 = (-b_hit + sqrt_disc) / (2 * a_hit)
        
        t_hit = np.where(t1 > 1e-6, t1, t2)
        valid = t_hit > 1e-6
        
        hit_indices = np.where(hit_mask)[0]
        valid_indices = hit_indices[valid]
        
        if len(valid_indices) == 0:
            return t_values, normals
        
        t_values[valid_indices] = t_hit[valid]
        hit_points = ray_origins[valid_indices] + t_hit[valid, np.newaxis] * ray_directions[valid_indices]
        normals[valid_indices] = (hit_points - self.position) / self.radius
        
        return t_values, normals
