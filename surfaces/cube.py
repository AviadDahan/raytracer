import numpy as np


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = np.array(position, dtype=np.float64)
        self.scale = scale
        self.material_index = material_index
        
        # Precompute min and max bounds (half the edge length in each direction)
        half_scale = scale / 2.0
        self.min_bound = self.position - half_scale
        self.max_bound = self.position + half_scale
    
    def intersect(self, ray_origin, ray_direction):
        """
        Compute ray-cube intersection using the slab method for axis-aligned box.
        
        Returns:
            (t, normal) if intersection found, where t > 0
            (None, None) if no intersection
        """
        t_min = -np.inf
        t_max = np.inf
        normal_axis = -1
        normal_sign = 1
        
        # Check each axis (x=0, y=1, z=2)
        for axis in range(3):
            if abs(ray_direction[axis]) < 1e-10:
                # Ray is parallel to the slab
                if ray_origin[axis] < self.min_bound[axis] or ray_origin[axis] > self.max_bound[axis]:
                    return None, None
            else:
                # Compute intersection t values with the two planes
                t1 = (self.min_bound[axis] - ray_origin[axis]) / ray_direction[axis]
                t2 = (self.max_bound[axis] - ray_origin[axis]) / ray_direction[axis]
                
                # Make t1 the near intersection, t2 the far
                if t1 > t2:
                    t1, t2 = t2, t1
                    sign = 1
                else:
                    sign = -1
                
                # Update the interval
                if t1 > t_min:
                    t_min = t1
                    normal_axis = axis
                    normal_sign = sign
                
                t_max = min(t_max, t2)
                
                # No intersection if interval is empty
                if t_min > t_max:
                    return None, None
        
        # Check if intersection is in front of ray origin
        if t_min < 1e-6:
            # Try the exit point
            if t_max < 1e-6:
                return None, None
            # We're inside the box, use exit point
            # For inside hits, we need to recalculate the normal
            t = t_max
            # Find which face we're exiting through
            for axis in range(3):
                t1 = (self.min_bound[axis] - ray_origin[axis]) / ray_direction[axis] if abs(ray_direction[axis]) > 1e-10 else np.inf
                t2 = (self.max_bound[axis] - ray_origin[axis]) / ray_direction[axis] if abs(ray_direction[axis]) > 1e-10 else np.inf
                if abs(max(t1, t2) - t_max) < 1e-6:
                    normal_axis = axis
                    normal_sign = 1 if t2 > t1 else -1
                    break
        else:
            t = t_min
        
        # Construct the normal
        normal = np.zeros(3)
        normal[normal_axis] = normal_sign
        
        return t, normal
