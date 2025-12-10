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
    
    def intersect_batch(self, ray_origins, ray_directions):
        """
        Compute ray-plane intersection for a batch of rays (vectorized).
        
        Args:
            ray_origins: (N, 3) array of ray origins
            ray_directions: (N, 3) array of ray directions (normalized)
        
        Returns:
            t_values: (N,) array of intersection distances (np.inf where no intersection)
            normals: (N, 3) array of surface normals at intersection points
        """
        N = ray_origins.shape[0]
        
        # denom = direction · normal for each ray: (N,)
        denom = np.dot(ray_directions, self.normal)
        
        # Initialize outputs
        t_values = np.full(N, np.inf)
        normals = np.zeros((N, 3))
        
        # Mask for non-parallel rays
        non_parallel = np.abs(denom) >= 1e-10
        
        if not np.any(non_parallel):
            return t_values, normals
        
        # Compute t for non-parallel rays
        # t = (offset - origin · normal) / denom
        origin_dot_normal = np.dot(ray_origins[non_parallel], self.normal)
        t_candidates = (self.offset - origin_dot_normal) / denom[non_parallel]
        
        # Valid if t > epsilon
        valid = t_candidates > 1e-6
        
        # Get indices
        non_parallel_indices = np.where(non_parallel)[0]
        valid_indices = non_parallel_indices[valid]
        
        if len(valid_indices) == 0:
            return t_values, normals
        
        # Store valid t values
        t_values[valid_indices] = t_candidates[valid]
        
        # Compute normals (pointing towards ray origin)
        # denom > 0 means ray going same direction as normal, so flip
        valid_denom = denom[valid_indices]
        normals[valid_indices] = np.where(
            valid_denom[:, np.newaxis] > 0,
            -self.normal,
            self.normal
        )
        
        return t_values, normals