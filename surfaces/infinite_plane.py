import numpy as np


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.offset = offset
        self.material_index = material_index
    
    def intersect(self, ray_origin, ray_direction):
        """Compute ray-plane intersection. Plane equation: P . N = offset"""
        denom = np.dot(ray_direction, self.normal)
        
        if abs(denom) < 1e-10:
            return None, None
        
        t = (self.offset - np.dot(ray_origin, self.normal)) / denom
        
        if t < 1e-6:
            return None, None
        
        normal = -self.normal if denom > 0 else self.normal
        return t, normal
    
    def intersect_batch(self, ray_origins, ray_directions):
        """Vectorized ray-plane intersection for N rays."""
        N = ray_origins.shape[0]
        
        denom = np.dot(ray_directions, self.normal)
        
        t_values = np.full(N, np.inf)
        normals = np.zeros((N, 3))
        
        non_parallel = np.abs(denom) >= 1e-10
        if not np.any(non_parallel):
            return t_values, normals
        
        origin_dot_normal = np.dot(ray_origins[non_parallel], self.normal)
        t_candidates = (self.offset - origin_dot_normal) / denom[non_parallel]
        
        valid = t_candidates > 1e-6
        non_parallel_indices = np.where(non_parallel)[0]
        valid_indices = non_parallel_indices[valid]
        
        if len(valid_indices) == 0:
            return t_values, normals
        
        t_values[valid_indices] = t_candidates[valid]
        
        valid_denom = denom[valid_indices]
        normals[valid_indices] = np.where(
            valid_denom[:, np.newaxis] > 0,
            -self.normal,
            self.normal
        )
        
        return t_values, normals
