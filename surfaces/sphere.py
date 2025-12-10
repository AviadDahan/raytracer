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
    
    def intersect_batch(self, ray_origins, ray_directions):
        """
        Compute ray-sphere intersection for a batch of rays (vectorized).
        
        Args:
            ray_origins: (N, 3) array of ray origins
            ray_directions: (N, 3) array of ray directions (normalized)
        
        Returns:
            t_values: (N,) array of intersection distances (np.inf where no intersection)
            normals: (N, 3) array of surface normals at intersection points
        """
        N = ray_origins.shape[0]
        
        # Vector from ray origins to sphere center: (N, 3)
        oc = ray_origins - self.position
        
        # Quadratic coefficients (vectorized): at² + bt + c = 0
        # a = dot(d, d) for each ray - should be 1 if normalized
        a = np.sum(ray_directions * ray_directions, axis=1)  # (N,)
        b = 2.0 * np.sum(oc * ray_directions, axis=1)        # (N,)
        c = np.sum(oc * oc, axis=1) - self.radius * self.radius  # (N,)
        
        discriminant = b * b - 4 * a * c  # (N,)
        
        # Initialize outputs
        t_values = np.full(N, np.inf)
        normals = np.zeros((N, 3))
        
        # Mask for rays that hit the sphere
        hit_mask = discriminant >= 0
        
        if not np.any(hit_mask):
            return t_values, normals
        
        # Compute t values only for hits
        sqrt_disc = np.sqrt(discriminant[hit_mask])
        a_hit = a[hit_mask]
        b_hit = b[hit_mask]
        
        t1 = (-b_hit - sqrt_disc) / (2 * a_hit)
        t2 = (-b_hit + sqrt_disc) / (2 * a_hit)
        
        # Choose smallest positive t
        # Start with t1, use t2 if t1 is invalid
        t_hit = np.where(t1 > 1e-6, t1, t2)
        
        # Create valid mask (positive t)
        valid = t_hit > 1e-6
        
        # Get indices of originally hit rays
        hit_indices = np.where(hit_mask)[0]
        valid_indices = hit_indices[valid]
        
        if len(valid_indices) == 0:
            return t_values, normals
        
        # Store valid t values
        t_values[valid_indices] = t_hit[valid]
        
        # Compute normals for valid hits
        hit_points = ray_origins[valid_indices] + t_hit[valid, np.newaxis] * ray_directions[valid_indices]
        normals[valid_indices] = (hit_points - self.position) / self.radius
        
        return t_values, normals