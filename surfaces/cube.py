import numpy as np


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = np.array(position, dtype=np.float64)
        self.scale = scale
        self.material_index = material_index
        
        half_scale = scale / 2.0
        self.min_bound = self.position - half_scale
        self.max_bound = self.position + half_scale
    
    def intersect(self, ray_origin, ray_direction):
        """Compute ray-cube intersection using the slab method."""
        t_min = -np.inf
        t_max = np.inf
        normal_axis = -1
        normal_sign = 1
        
        for axis in range(3):
            if abs(ray_direction[axis]) < 1e-10:
                if ray_origin[axis] < self.min_bound[axis] or ray_origin[axis] > self.max_bound[axis]:
                    return None, None
            else:
                t1 = (self.min_bound[axis] - ray_origin[axis]) / ray_direction[axis]
                t2 = (self.max_bound[axis] - ray_origin[axis]) / ray_direction[axis]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                    sign = 1
                else:
                    sign = -1
                
                if t1 > t_min:
                    t_min = t1
                    normal_axis = axis
                    normal_sign = sign
                
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    return None, None
        
        if t_min < 1e-6:
            if t_max < 1e-6:
                return None, None
            t = t_max
            for axis in range(3):
                t1 = (self.min_bound[axis] - ray_origin[axis]) / ray_direction[axis] if abs(ray_direction[axis]) > 1e-10 else np.inf
                t2 = (self.max_bound[axis] - ray_origin[axis]) / ray_direction[axis] if abs(ray_direction[axis]) > 1e-10 else np.inf
                if abs(max(t1, t2) - t_max) < 1e-6:
                    normal_axis = axis
                    normal_sign = 1 if t2 > t1 else -1
                    break
        else:
            t = t_min
        
        normal = np.zeros(3)
        normal[normal_axis] = normal_sign
        
        return t, normal
    
    def intersect_batch(self, ray_origins, ray_directions):
        """Vectorized ray-cube intersection using the slab method."""
        N = ray_origins.shape[0]
        
        t_values = np.full(N, np.inf)
        normals = np.zeros((N, 3))
        t_min = np.full(N, -np.inf)
        t_max = np.full(N, np.inf)
        normal_axis = np.zeros(N, dtype=np.int32)
        normal_sign = np.ones(N)
        valid = np.ones(N, dtype=bool)
        
        for axis in range(3):
            d = ray_directions[:, axis]
            o = ray_origins[:, axis]
            
            parallel = np.abs(d) < 1e-10
            outside = (o < self.min_bound[axis]) | (o > self.max_bound[axis])
            valid[parallel & outside] = False
            
            non_parallel = ~parallel & valid
            if not np.any(non_parallel):
                continue
            
            d_np = d[non_parallel]
            o_np = o[non_parallel]
            
            t1 = (self.min_bound[axis] - o_np) / d_np
            t2 = (self.max_bound[axis] - o_np) / d_np
            
            sign = np.where(t1 > t2, 1.0, -1.0)
            t1_near = np.minimum(t1, t2)
            t2_far = np.maximum(t1, t2)
            
            np_indices = np.where(non_parallel)[0]
            update_tmin = t1_near > t_min[non_parallel]
            update_indices = np_indices[update_tmin]
            
            t_min[update_indices] = t1_near[update_tmin]
            normal_axis[update_indices] = axis
            normal_sign[update_indices] = sign[update_tmin]
            
            t_max[non_parallel] = np.minimum(t_max[non_parallel], t2_far)
            valid[non_parallel] = valid[non_parallel] & (t_min[non_parallel] <= t_max[non_parallel])
        
        if not np.any(valid):
            return t_values, normals
        
        inside = (t_min < 1e-6) & valid
        outside_front = (t_min >= 1e-6) & valid
        
        if np.any(outside_front):
            t_values[outside_front] = t_min[outside_front]
            for i in np.where(outside_front)[0]:
                normals[i, normal_axis[i]] = normal_sign[i]
        
        inside_valid = inside & (t_max >= 1e-6)
        if np.any(inside_valid):
            t_values[inside_valid] = t_max[inside_valid]
            for i in np.where(inside_valid)[0]:
                d = ray_directions[i]
                o = ray_origins[i]
                t = t_max[i]
                for axis in range(3):
                    if abs(d[axis]) < 1e-10:
                        continue
                    t1 = (self.min_bound[axis] - o[axis]) / d[axis]
                    t2 = (self.max_bound[axis] - o[axis]) / d[axis]
                    if abs(max(t1, t2) - t) < 1e-6:
                        normals[i, axis] = 1.0 if t2 > t1 else -1.0
                        break
        
        return t_values, normals
