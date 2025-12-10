import argparse
from PIL import Image
import numpy as np
from numba import njit

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


# Small epsilon to avoid self-intersection
EPSILON = 1e-6


# =============================================================================
# Numba JIT-compiled helper functions for hot paths
# =============================================================================

@njit(cache=True)
def _get_material_indices(surface_material_indices, hit_surf_indices):
    """
    Get material indices for hit surfaces (JIT-compiled).
    Replaces: np.array([surfaces[si].material_index - 1 for si in hit_surf_idx])
    """
    n = len(hit_surf_indices)
    result = np.empty(n, dtype=np.int32)
    for i in range(n):
        result[i] = surface_material_indices[hit_surf_indices[i]]
    return result


@njit(cache=True)
def _sphere_intersect_batch(ray_origins, ray_directions, center, radius):
    """
    Vectorized sphere intersection (JIT-compiled for speed).
    """
    N = ray_origins.shape[0]
    t_values = np.full(N, np.inf)
    normals = np.zeros((N, 3))
    
    for i in range(N):
        # Vector from ray origin to sphere center
        oc_x = ray_origins[i, 0] - center[0]
        oc_y = ray_origins[i, 1] - center[1]
        oc_z = ray_origins[i, 2] - center[2]
        
        dx = ray_directions[i, 0]
        dy = ray_directions[i, 1]
        dz = ray_directions[i, 2]
        
        # Quadratic coefficients
        a = dx*dx + dy*dy + dz*dz
        b = 2.0 * (oc_x*dx + oc_y*dy + oc_z*dz)
        c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            
            if t1 > 1e-6:
                t = t1
            elif t2 > 1e-6:
                t = t2
            else:
                continue
            
            t_values[i] = t
            
            # Compute hit point and normal
            hit_x = ray_origins[i, 0] + t * dx
            hit_y = ray_origins[i, 1] + t * dy
            hit_z = ray_origins[i, 2] + t * dz
            
            normals[i, 0] = (hit_x - center[0]) / radius
            normals[i, 1] = (hit_y - center[1]) / radius
            normals[i, 2] = (hit_z - center[2]) / radius
    
    return t_values, normals


@njit(cache=True)
def _plane_intersect_batch(ray_origins, ray_directions, normal, offset):
    """
    Vectorized plane intersection (JIT-compiled for speed).
    """
    N = ray_origins.shape[0]
    t_values = np.full(N, np.inf)
    normals_out = np.zeros((N, 3))
    
    for i in range(N):
        # denom = direction · normal
        denom = (ray_directions[i, 0] * normal[0] + 
                 ray_directions[i, 1] * normal[1] + 
                 ray_directions[i, 2] * normal[2])
        
        if abs(denom) < 1e-10:
            continue
        
        # t = (offset - origin · normal) / denom
        origin_dot_normal = (ray_origins[i, 0] * normal[0] + 
                            ray_origins[i, 1] * normal[1] + 
                            ray_origins[i, 2] * normal[2])
        t = (offset - origin_dot_normal) / denom
        
        if t > 1e-6:
            t_values[i] = t
            # Normal points towards ray origin
            if denom > 0:
                normals_out[i, 0] = -normal[0]
                normals_out[i, 1] = -normal[1]
                normals_out[i, 2] = -normal[2]
            else:
                normals_out[i, 0] = normal[0]
                normals_out[i, 1] = normal[1]
                normals_out[i, 2] = normal[2]
    
    return t_values, normals_out


@njit(cache=True)
def _cube_intersect_batch(ray_origins, ray_directions, min_bound, max_bound):
    """
    Vectorized cube intersection using slab method (JIT-compiled).
    """
    N = ray_origins.shape[0]
    t_values = np.full(N, np.inf)
    normals = np.zeros((N, 3))
    
    for i in range(N):
        t_min = -np.inf
        t_max = np.inf
        normal_axis = 0
        normal_sign = 1.0
        valid = True
        
        for axis in range(3):
            d = ray_directions[i, axis]
            o = ray_origins[i, axis]
            
            if abs(d) < 1e-10:
                # Ray parallel to slab
                if o < min_bound[axis] or o > max_bound[axis]:
                    valid = False
                    break
            else:
                t1 = (min_bound[axis] - o) / d
                t2 = (max_bound[axis] - o) / d
                
                if t1 > t2:
                    t1, t2 = t2, t1
                    sign = 1.0
                else:
                    sign = -1.0
                
                if t1 > t_min:
                    t_min = t1
                    normal_axis = axis
                    normal_sign = sign
                
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    valid = False
                    break
        
        if not valid:
            continue
        
        if t_min >= 1e-6:
            t_values[i] = t_min
            normals[i, normal_axis] = normal_sign
        elif t_max >= 1e-6:
            t_values[i] = t_max
            # Find exit face for inside hits
            for axis in range(3):
                d = ray_directions[i, axis]
                if abs(d) < 1e-10:
                    continue
                o = ray_origins[i, axis]
                t1 = (min_bound[axis] - o) / d
                t2 = (max_bound[axis] - o) / d
                if abs(max(t1, t2) - t_max) < 1e-6:
                    if t2 > t1:
                        normals[i, axis] = 1.0
                    else:
                        normals[i, axis] = -1.0
                    break
    
    return t_values, normals


@njit(cache=True)
def _find_nearest_intersection_jit(ray_origins, ray_directions, 
                                   sphere_centers, sphere_radii,
                                   plane_normals, plane_offsets,
                                   cube_min_bounds, cube_max_bounds,
                                   surface_types, surface_indices,
                                   num_spheres, num_planes, num_cubes):
    """
    Find nearest intersection for all rays against all surfaces (JIT-accelerated).
    
    surface_types: 0=sphere, 1=plane, 2=cube
    surface_indices: index into respective type array
    
    Note: We use regular range instead of prange to avoid thread contention
    when used with multiprocessing. The parallelization is handled at the 
    process level by render_parallel.
    """
    N = ray_origins.shape[0]
    num_surfaces = len(surface_types)
    
    best_t = np.full(N, np.inf)
    best_surf_idx = np.full(N, -1, dtype=np.int32)
    best_normals = np.zeros((N, 3))
    
    for ray_idx in range(N):
        ray_o = ray_origins[ray_idx]
        ray_d = ray_directions[ray_idx]
        
        for surf_idx in range(num_surfaces):
            surf_type = surface_types[surf_idx]
            type_idx = surface_indices[surf_idx]
            
            t = np.inf
            nx, ny, nz = 0.0, 0.0, 0.0
            
            if surf_type == 0:  # Sphere
                center = sphere_centers[type_idx]
                radius = sphere_radii[type_idx]
                
                oc_x = ray_o[0] - center[0]
                oc_y = ray_o[1] - center[1]
                oc_z = ray_o[2] - center[2]
                
                a = ray_d[0]**2 + ray_d[1]**2 + ray_d[2]**2
                b = 2.0 * (oc_x*ray_d[0] + oc_y*ray_d[1] + oc_z*ray_d[2])
                c = oc_x**2 + oc_y**2 + oc_z**2 - radius**2
                
                disc = b*b - 4*a*c
                if disc >= 0:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-b - sqrt_disc) / (2*a)
                    t2 = (-b + sqrt_disc) / (2*a)
                    
                    if t1 > 1e-6:
                        t = t1
                    elif t2 > 1e-6:
                        t = t2
                    
                    if t < np.inf:
                        hit_x = ray_o[0] + t * ray_d[0]
                        hit_y = ray_o[1] + t * ray_d[1]
                        hit_z = ray_o[2] + t * ray_d[2]
                        nx = (hit_x - center[0]) / radius
                        ny = (hit_y - center[1]) / radius
                        nz = (hit_z - center[2]) / radius
            
            elif surf_type == 1:  # Plane
                normal = plane_normals[type_idx]
                offset = plane_offsets[type_idx]
                
                denom = ray_d[0]*normal[0] + ray_d[1]*normal[1] + ray_d[2]*normal[2]
                if abs(denom) >= 1e-10:
                    origin_dot = ray_o[0]*normal[0] + ray_o[1]*normal[1] + ray_o[2]*normal[2]
                    t_cand = (offset - origin_dot) / denom
                    if t_cand > 1e-6:
                        t = t_cand
                        if denom > 0:
                            nx, ny, nz = -normal[0], -normal[1], -normal[2]
                        else:
                            nx, ny, nz = normal[0], normal[1], normal[2]
            
            elif surf_type == 2:  # Cube
                min_b = cube_min_bounds[type_idx]
                max_b = cube_max_bounds[type_idx]
                
                t_min = -np.inf
                t_max_local = np.inf
                n_axis = 0
                n_sign = 1.0
                valid = True
                
                for axis in range(3):
                    d = ray_d[axis]
                    o = ray_o[axis]
                    
                    if abs(d) < 1e-10:
                        if o < min_b[axis] or o > max_b[axis]:
                            valid = False
                            break
                    else:
                        t1 = (min_b[axis] - o) / d
                        t2 = (max_b[axis] - o) / d
                        
                        if t1 > t2:
                            t1, t2 = t2, t1
                            sign = 1.0
                        else:
                            sign = -1.0
                        
                        if t1 > t_min:
                            t_min = t1
                            n_axis = axis
                            n_sign = sign
                        
                        if t2 < t_max_local:
                            t_max_local = t2
                        
                        if t_min > t_max_local:
                            valid = False
                            break
                
                if valid:
                    if t_min >= 1e-6:
                        t = t_min
                        if n_axis == 0:
                            nx = n_sign
                        elif n_axis == 1:
                            ny = n_sign
                        else:
                            nz = n_sign
                    elif t_max_local >= 1e-6:
                        t = t_max_local
            
            if t < best_t[ray_idx]:
                best_t[ray_idx] = t
                best_surf_idx[ray_idx] = surf_idx
                best_normals[ray_idx, 0] = nx
                best_normals[ray_idx, 1] = ny
                best_normals[ray_idx, 2] = nz
    
    return best_t, best_surf_idx, best_normals


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    return v / norm


def normalize_batch(v):
    """Normalize an array of vectors (N, 3)."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)  # Avoid division by zero
    return v / norms


def reflect(d, n):
    """Reflect direction d around normal n."""
    return d - 2 * np.dot(d, n) * n


def reflect_batch(d, n):
    """Reflect directions d around normals n. Both are (N, 3) arrays."""
    # d - 2 * (d · n) * n
    dot = np.sum(d * n, axis=1, keepdims=True)  # (N, 1)
    return d - 2 * dot * n


def parse_scene_file(file_path):
    """Parse the scene file and return camera, settings, and scene objects."""
    objects = []
    camera = None
    scene_settings = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    
    return camera, scene_settings, objects


def separate_objects(objects):
    """Separate parsed objects into materials, surfaces, and lights."""
    materials = []
    surfaces = []
    lights = []
    
    for obj in objects:
        if isinstance(obj, Material):
            materials.append(obj)
        elif isinstance(obj, Light):
            lights.append(obj)
        elif isinstance(obj, (Sphere, InfinitePlane, Cube)):
            surfaces.append(obj)
    
    return materials, surfaces, lights


def find_nearest_intersection(ray_origin, ray_direction, surfaces, max_t=np.inf):
    """
    Find the nearest surface intersection along the ray.
    
    Returns:
        (t, surface, normal) if intersection found
        (None, None, None) if no intersection
    """
    nearest_t = max_t
    nearest_surface = None
    nearest_normal = None
    
    for surface in surfaces:
        t, normal = surface.intersect(ray_origin, ray_direction)
        if t is not None and t < nearest_t:
            nearest_t = t
            nearest_surface = surface
            nearest_normal = normal
    
    if nearest_surface is None:
        return None, None, None
    
    return nearest_t, nearest_surface, nearest_normal


def find_nearest_intersection_batch(ray_origins, ray_directions, surfaces, max_t=np.inf):
    """
    Find the nearest surface intersection for a batch of rays (vectorized).
    
    Args:
        ray_origins: (N, 3) array of ray origins
        ray_directions: (N, 3) array of ray directions
        surfaces: list of surface objects
        max_t: maximum t value to consider
    
    Returns:
        t_values: (N,) array of intersection distances (np.inf where no hit)
        surface_indices: (N,) array of surface indices (-1 where no hit)
        normals: (N, 3) array of surface normals
    """
    N = ray_origins.shape[0]
    
    # Initialize with no hits
    best_t = np.full(N, max_t if max_t != np.inf else np.inf)
    best_surface_idx = np.full(N, -1, dtype=np.int32)
    best_normals = np.zeros((N, 3))
    
    # Test each surface
    for surf_idx, surface in enumerate(surfaces):
        t_values, normals = surface.intersect_batch(ray_origins, ray_directions)
        
        # Update where this surface is closer
        closer = t_values < best_t
        best_t[closer] = t_values[closer]
        best_surface_idx[closer] = surf_idx
        best_normals[closer] = normals[closer]
    
    return best_t, best_surface_idx, best_normals


def prepare_surface_data(surfaces):
    """
    Prepare surface data as numpy arrays for JIT-compiled intersection.
    
    Returns a dict containing arrays that can be passed to _find_nearest_intersection_jit.
    """
    spheres = [(i, s) for i, s in enumerate(surfaces) if isinstance(s, Sphere)]
    planes = [(i, s) for i, s in enumerate(surfaces) if isinstance(s, InfinitePlane)]
    cubes = [(i, s) for i, s in enumerate(surfaces) if isinstance(s, Cube)]
    
    num_spheres = len(spheres)
    num_planes = len(planes)
    num_cubes = len(cubes)
    num_surfaces = len(surfaces)
    
    # Arrays for spheres
    sphere_centers = np.zeros((max(1, num_spheres), 3))
    sphere_radii = np.zeros(max(1, num_spheres))
    for idx, (_, s) in enumerate(spheres):
        sphere_centers[idx] = s.position
        sphere_radii[idx] = s.radius
    
    # Arrays for planes
    plane_normals = np.zeros((max(1, num_planes), 3))
    plane_offsets = np.zeros(max(1, num_planes))
    for idx, (_, s) in enumerate(planes):
        plane_normals[idx] = s.normal
        plane_offsets[idx] = s.offset
    
    # Arrays for cubes
    cube_min_bounds = np.zeros((max(1, num_cubes), 3))
    cube_max_bounds = np.zeros((max(1, num_cubes), 3))
    for idx, (_, s) in enumerate(cubes):
        cube_min_bounds[idx] = s.min_bound
        cube_max_bounds[idx] = s.max_bound
    
    # Surface type mapping: 0=sphere, 1=plane, 2=cube
    surface_types = np.zeros(num_surfaces, dtype=np.int32)
    surface_type_indices = np.zeros(num_surfaces, dtype=np.int32)
    
    sphere_idx = 0
    plane_idx = 0
    cube_idx = 0
    for i, s in enumerate(surfaces):
        if isinstance(s, Sphere):
            surface_types[i] = 0
            surface_type_indices[i] = sphere_idx
            sphere_idx += 1
        elif isinstance(s, InfinitePlane):
            surface_types[i] = 1
            surface_type_indices[i] = plane_idx
            plane_idx += 1
        elif isinstance(s, Cube):
            surface_types[i] = 2
            surface_type_indices[i] = cube_idx
            cube_idx += 1
    
    # Pre-compute material indices (0-indexed)
    material_indices = np.array([s.material_index - 1 for s in surfaces], dtype=np.int32)
    
    return {
        'sphere_centers': sphere_centers,
        'sphere_radii': sphere_radii,
        'plane_normals': plane_normals,
        'plane_offsets': plane_offsets,
        'cube_min_bounds': cube_min_bounds,
        'cube_max_bounds': cube_max_bounds,
        'surface_types': surface_types,
        'surface_indices': surface_type_indices,
        'material_indices': material_indices,
        'num_spheres': num_spheres,
        'num_planes': num_planes,
        'num_cubes': num_cubes,
    }


def find_nearest_intersection_batch_jit(ray_origins, ray_directions, surface_data, max_t=np.inf):
    """
    JIT-accelerated version of find_nearest_intersection_batch.
    Uses pre-computed surface data arrays.
    """
    t_values, surface_indices, normals = _find_nearest_intersection_jit(
        ray_origins, ray_directions,
        surface_data['sphere_centers'], surface_data['sphere_radii'],
        surface_data['plane_normals'], surface_data['plane_offsets'],
        surface_data['cube_min_bounds'], surface_data['cube_max_bounds'],
        surface_data['surface_types'], surface_data['surface_indices'],
        surface_data['num_spheres'], surface_data['num_planes'], surface_data['num_cubes']
    )
    
    # Apply max_t filter
    if max_t < np.inf:
        too_far = t_values > max_t
        surface_indices[too_far] = -1
    
    return t_values, surface_indices, normals


def compute_soft_shadow(hit_point, light, surfaces, materials, n_shadow_rays):
    """
    Compute soft shadow with transparency (bonus feature).
    
    Returns the light intensity factor (0 to 1).
    """
    light_pos = np.array(light.position, dtype=np.float64)
    
    # Direction from hit point to light
    to_light = light_pos - hit_point
    distance_to_light = np.linalg.norm(to_light)
    light_dir = to_light / distance_to_light
    
    if n_shadow_rays <= 1:
        # Single shadow ray (hard shadow)
        transmission = compute_shadow_ray_transmission(
            hit_point, light_pos, distance_to_light, surfaces, materials
        )
        return transmission
    
    # Construct a plane perpendicular to the light direction
    # Find two perpendicular vectors on this plane
    if abs(light_dir[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    
    right = normalize(np.cross(light_dir, up))
    up = normalize(np.cross(right, light_dir))
    
    # Grid cell size
    cell_size = light.radius / n_shadow_rays
    
    total_transmission = 0.0
    
    # Sample N×N grid
    for i in range(n_shadow_rays):
        for j in range(n_shadow_rays):
            # Random point within the cell
            # Cell ranges from -radius/2 to +radius/2
            offset_i = -light.radius / 2 + cell_size * (i + np.random.random())
            offset_j = -light.radius / 2 + cell_size * (j + np.random.random())
            
            # Sample point position
            sample_pos = light_pos + right * offset_i + up * offset_j
            
            # Compute transmission for this shadow ray
            to_sample = sample_pos - hit_point
            dist_to_sample = np.linalg.norm(to_sample)
            
            transmission = compute_shadow_ray_transmission(
                hit_point, sample_pos, dist_to_sample, surfaces, materials
            )
            total_transmission += transmission
    
    # Average transmission
    return total_transmission / (n_shadow_rays * n_shadow_rays)


def compute_soft_shadow_batch(hit_points, hit_mask, light, surface_data, material_indices, materials_array, n_shadow_rays):
    """
    Compute soft shadows for a batch of hit points (vectorized).
    
    Args:
        hit_points: (N, 3) array of hit points
        hit_mask: (N,) boolean mask of valid hit points
        light: Light object
        surface_data: pre-computed surface data for JIT intersection
        material_indices: (N,) array of material indices for each hit point
        materials_array: dict with material properties as arrays
        n_shadow_rays: number of shadow rays per dimension
    
    Returns:
        transmissions: (N,) array of transmission values (0 to 1)
    """
    N = hit_points.shape[0]
    transmissions = np.ones(N)  # Default to fully lit
    
    if not np.any(hit_mask):
        return transmissions
    
    light_pos = np.array(light.position, dtype=np.float64)
    
    # Get active hit points
    active_indices = np.where(hit_mask)[0]
    active_points = hit_points[active_indices]
    M = len(active_indices)
    
    if n_shadow_rays <= 1:
        # Hard shadows - single ray per point
        to_light = light_pos - active_points  # (M, 3)
        distances = np.linalg.norm(to_light, axis=1)  # (M,)
        directions = to_light / distances[:, np.newaxis]  # (M, 3)
        
        # Offset origins slightly
        shadow_origins = active_points + directions * EPSILON
        
        trans = compute_shadow_transmission_batch(
            shadow_origins, directions, distances - EPSILON, 
            surface_data, material_indices[active_indices], materials_array
        )
        transmissions[active_indices] = trans
        return transmissions
    
    # Soft shadows - sample NxN grid on light plane
    # Compute per-point light plane basis vectors (M, 3) each
    to_light = light_pos - active_points  # (M, 3)
    light_dists = np.linalg.norm(to_light, axis=1, keepdims=True)  # (M, 1)
    light_dirs = to_light / light_dists  # (M, 3)
    
    # Choose up reference based on light direction to avoid degeneracy
    # For each point, use [1,0,0] unless light_dir is nearly parallel to it
    up_ref = np.zeros((M, 3))
    use_y = np.abs(light_dirs[:, 0]) >= 0.9
    up_ref[~use_y] = [1.0, 0.0, 0.0]
    up_ref[use_y] = [0.0, 1.0, 0.0]
    
    # Compute right and up vectors for each point
    # right = normalize(cross(light_dir, up_ref))
    right = np.cross(light_dirs, up_ref)  # (M, 3)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    right = right / np.maximum(right_norms, EPSILON)
    
    # up = normalize(cross(right, light_dir))
    up = np.cross(right, light_dirs)  # (M, 3)
    up_norms = np.linalg.norm(up, axis=1, keepdims=True)
    up = up / np.maximum(up_norms, EPSILON)
    
    cell_size = light.radius / n_shadow_rays
    total_samples = n_shadow_rays * n_shadow_rays
    
    # Accumulate transmissions
    total_trans = np.zeros(M)
    
    for i in range(n_shadow_rays):
        for j in range(n_shadow_rays):
            # Random offset within cell (vectorized random)
            rand_i = np.random.random(M)
            rand_j = np.random.random(M)
            
            offset_i = -light.radius / 2 + cell_size * (i + rand_i)  # (M,)
            offset_j = -light.radius / 2 + cell_size * (j + rand_j)  # (M,)
            
            # Sample positions: per-point basis vectors
            # sample_pos[k] = light_pos + offset_i[k] * right[k] + offset_j[k] * up[k]
            sample_pos = (light_pos + 
                         offset_i[:, np.newaxis] * right + 
                         offset_j[:, np.newaxis] * up)  # (M, 3)
            
            # Shadow ray directions
            to_sample = sample_pos - active_points
            distances = np.linalg.norm(to_sample, axis=1)
            directions = to_sample / distances[:, np.newaxis]
            
            shadow_origins = active_points + directions * EPSILON
            
            trans = compute_shadow_transmission_batch(
                shadow_origins, directions, distances - EPSILON,
                surface_data, material_indices[active_indices], materials_array
            )
            total_trans += trans
    
    transmissions[active_indices] = total_trans / total_samples
    return transmissions


def compute_shadow_transmission_batch(origins, directions, max_distances, surface_data, 
                                      hit_material_indices, materials_array):
    """
    Compute shadow ray transmission for a batch of rays, accounting for transparency.
    
    Args:
        origins: (M, 3) shadow ray origins
        directions: (M, 3) shadow ray directions
        max_distances: (M,) max distance to light
        surface_data: pre-computed surface data for JIT intersection
        hit_material_indices: (M,) material indices of the primary hit surfaces
        materials_array: dict with transparency array
    
    Returns:
        transmissions: (M,) transmission values
    """
    M = origins.shape[0]
    transmissions = np.ones(M)
    active = np.ones(M, dtype=bool)
    
    current_origins = origins.copy()
    remaining_distances = max_distances.copy()
    
    # Iterate until all rays reach light or are blocked
    max_iterations = 20  # Prevent infinite loops
    for _ in range(max_iterations):
        if not np.any(active):
            break
        
        # Find intersections for active rays (using JIT version)
        active_indices = np.where(active)[0]
        t_vals, surf_idx, _ = find_nearest_intersection_batch_jit(
            current_origins[active_indices],
            directions[active_indices],
            surface_data
        )
        
        # Check which rays hit something before the light
        hit_before_light = (t_vals < remaining_distances[active_indices]) & (surf_idx >= 0)
        
        if not np.any(hit_before_light):
            break
        
        # Get material transparency for blocking surfaces
        hit_idx = active_indices[hit_before_light]
        blocking_surf_idx = surf_idx[hit_before_light]
        
        # Get material index for each blocking surface (using pre-computed array)
        blocking_mat_idx = surface_data['material_indices'][blocking_surf_idx]
        
        # Multiply transmission by transparency
        transparencies = materials_array['transparency'][blocking_mat_idx]
        transmissions[hit_idx] *= transparencies
        
        # Deactivate rays with negligible transmission
        active[hit_idx[transmissions[hit_idx] < EPSILON]] = False
        
        # Move past the blocking surface
        still_active_in_hit = transmissions[hit_idx] >= EPSILON
        if np.any(still_active_in_hit):
            move_idx = hit_idx[still_active_in_hit]
            move_t = t_vals[hit_before_light][still_active_in_hit]
            current_origins[move_idx] += directions[move_idx] * (move_t[:, np.newaxis] + EPSILON)
            remaining_distances[move_idx] -= (move_t + EPSILON)
        
        # Deactivate rays that didn't hit anything before light
        no_hit_idx = active_indices[~hit_before_light]
        active[no_hit_idx] = False
    
    return transmissions


def compute_shadow_ray_transmission(origin, light_pos, max_distance, surfaces, materials):
    """
    Compute the transmission along a shadow ray, accounting for transparent objects.
    
    Returns a value from 0 (fully blocked) to 1 (fully lit).
    """
    direction = (light_pos - origin) / max_distance
    
    # Start slightly offset to avoid self-intersection
    current_origin = origin + direction * EPSILON
    remaining_distance = max_distance - EPSILON
    
    transmission = 1.0
    
    while remaining_distance > EPSILON:
        t, surface, _ = find_nearest_intersection(
            current_origin, direction, surfaces, remaining_distance
        )
        
        if t is None:
            # No more intersections, light reaches the point
            break
        
        # Get the material of the blocking object
        material = materials[surface.material_index - 1]  # 1-indexed
        
        # Multiply transmission by the object's transparency
        transmission *= material.transparency
        
        if transmission < EPSILON:
            # Fully blocked
            return 0.0
        
        # Move past this object
        current_origin = current_origin + direction * (t + EPSILON)
        remaining_distance -= (t + EPSILON)
    
    return transmission


def compute_color(ray_origin, ray_direction, hit_point, normal, surface, 
                  materials, surfaces, lights, scene_settings, depth):
    """
    Compute the color at a hit point using Phong shading model.
    """
    material = materials[surface.material_index - 1]  # 1-indexed
    
    background_color = np.array(scene_settings.background_color, dtype=np.float64)
    n_shadow_rays = int(scene_settings.root_number_shadow_rays)
    
    # View direction (from hit point to camera)
    view_dir = normalize(ray_origin - hit_point)
    
    # Initialize diffuse and specular components
    diffuse = np.zeros(3)
    specular = np.zeros(3)
    
    # Compute lighting from each light source
    for light in lights:
        light_pos = np.array(light.position, dtype=np.float64)
        light_color = np.array(light.color, dtype=np.float64)
        
        # Direction from hit point to light
        light_dir = normalize(light_pos - hit_point)
        
        # Compute soft shadow with transparency
        shadow_transmission = compute_soft_shadow(
            hit_point + normal * EPSILON, light, surfaces, materials, n_shadow_rays
        )
        
        # Light intensity formula from assignment:
        # light_intensity = (1 - shadow_intensity) + shadow_intensity * transmission
        light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * shadow_transmission
        
        if light_intensity < EPSILON:
            continue
        
        # Diffuse component: material.diffuse * light.color * max(0, N·L)
        n_dot_l = max(0.0, np.dot(normal, light_dir))
        diffuse += (np.array(material.diffuse_color) * light_color * 
                   light_intensity * n_dot_l)
        
        # Specular component (Phong): material.specular * light.color * 
        #                             spec_intensity * max(0, R·V)^shininess
        # R is the reflection of light direction around normal
        reflect_dir = reflect(-light_dir, normal)
        r_dot_v = max(0.0, np.dot(reflect_dir, view_dir))
        
        if r_dot_v > 0:
            specular += (np.array(material.specular_color) * light_color * 
                        light.specular_intensity * light_intensity * 
                        (r_dot_v ** material.shininess))
    
    # Reflection color
    reflection_color = np.zeros(3)
    if depth > 0 and np.any(np.array(material.reflection_color) > EPSILON):
        # Compute reflection direction
        reflect_ray_dir = reflect(ray_direction, normal)
        reflect_origin = hit_point + normal * EPSILON
        
        # Recursively trace reflection ray
        reflected = trace_ray(reflect_origin, reflect_ray_dir, 
                            materials, surfaces, lights, scene_settings, depth - 1)
        reflection_color = np.array(material.reflection_color) * reflected
    
    # Background color (for transparency)
    bg_color = background_color
    if material.transparency > EPSILON and depth > 0:
        # Continue ray through the surface
        # Offset in the direction of the ray (through the surface)
        continue_origin = hit_point - normal * EPSILON
        bg_color = trace_ray(continue_origin, ray_direction,
                           materials, surfaces, lights, scene_settings, depth - 1)
    
    # Final color formula from assignment:
    # output_color = background_color * transparency 
    #              + (diffuse + specular) * (1 - transparency) 
    #              + reflection_color
    output_color = (bg_color * material.transparency + 
                   (diffuse + specular) * (1 - material.transparency) + 
                   reflection_color)
    
    return output_color


def trace_ray(ray_origin, ray_direction, materials, surfaces, lights, scene_settings, depth):
    """
    Trace a ray through the scene and return the color.
    """
    if depth < 0:
        return np.array(scene_settings.background_color, dtype=np.float64)
    
    # Find nearest intersection
    t, surface, normal = find_nearest_intersection(ray_origin, ray_direction, surfaces)
    
    if surface is None:
        return np.array(scene_settings.background_color, dtype=np.float64)
    
    # Compute hit point
    hit_point = ray_origin + t * ray_direction
    
    # Compute color at hit point
    return compute_color(ray_origin, ray_direction, hit_point, normal, surface,
                        materials, surfaces, lights, scene_settings, depth)


def render(camera, scene_settings, materials, surfaces, lights, width, height):
    """
    Render the scene to an image array (original sequential version).
    """
    import sys
    import time
    
    # Setup camera
    aspect_ratio = width / height
    camera.setup(aspect_ratio)
    print(f"Camera setup complete. Forward: {camera.forward}, Right: {camera.right}, Up: {camera.up}")
    
    # Pre-allocate image array
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    max_depth = int(scene_settings.max_recursions)
    n_shadow = int(scene_settings.root_number_shadow_rays)
    print(f"Max depth: {max_depth}, Shadow rays: {n_shadow}x{n_shadow}={n_shadow*n_shadow} per light")
    
    # Render each pixel
    total_pixels = width * height
    start_time = time.time()
    
    for y in range(height):
        row_start = time.time()
        for x in range(width):
            # Generate ray for this pixel
            ray_origin, ray_direction = camera.generate_ray(x, y, width, height)
            
            # Trace the ray
            color = trace_ray(ray_origin, ray_direction, 
                            materials, surfaces, lights, scene_settings, max_depth)
            
            image[y, x] = color
        
        # Progress indicator every 10 rows
        if (y + 1) % 10 == 0 or y == height - 1:
            elapsed = time.time() - start_time
            progress = (y + 1) / height
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            row_time = time.time() - row_start
            print(f"Row {y+1}/{height} ({progress*100:.1f}%) - Row time: {row_time:.2f}s - ETA: {eta:.0f}s")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    print(f"Rendering complete in {total_time:.1f}s")
    
    return image


def render_vectorized(camera, scene_settings, materials, surfaces, lights, width, height):
    """
    Render the scene using vectorized operations (much faster).
    """
    import sys
    import time
    
    start_time = time.time()
    
    # Setup camera
    aspect_ratio = width / height
    camera.setup(aspect_ratio)
    print(f"Camera setup complete. Forward: {camera.forward}, Right: {camera.right}, Up: {camera.up}")
    
    max_depth = int(scene_settings.max_recursions)
    n_shadow = int(scene_settings.root_number_shadow_rays)
    background_color = np.array(scene_settings.background_color, dtype=np.float64)
    
    print(f"Max depth: {max_depth}, Shadow rays: {n_shadow}x{n_shadow}={n_shadow*n_shadow} per light")
    print(f"Vectorized rendering {width}x{height} = {width*height} rays...")
    
    # Pre-compute material properties as arrays for fast lookup
    num_materials = len(materials)
    materials_array = {
        'diffuse': np.array([m.diffuse_color for m in materials]),       # (num_mat, 3)
        'specular': np.array([m.specular_color for m in materials]),     # (num_mat, 3)
        'reflection': np.array([m.reflection_color for m in materials]), # (num_mat, 3)
        'shininess': np.array([m.shininess for m in materials]),         # (num_mat,)
        'transparency': np.array([m.transparency for m in materials]),   # (num_mat,)
    }
    
    # Prepare surface data for JIT-accelerated intersection (one-time cost)
    surface_data = prepare_surface_data(surfaces)
    print(f"Prepared surface data for JIT: {len(surfaces)} surfaces")
    
    # Generate all primary rays at once
    ray_gen_start = time.time()
    ray_origins, ray_directions = camera.generate_all_rays(width, height)
    N = ray_origins.shape[0]
    print(f"Generated {N} rays in {time.time() - ray_gen_start:.3f}s")
    
    # Initialize output colors to zero (will accumulate)
    colors = np.zeros((N, 3))
    
    # Initialize ray contribution weights (for recursive rays)
    weights = np.ones((N, 3))
    
    # Current rays being traced
    current_origins = ray_origins.copy()
    current_directions = ray_directions.copy()
    active = np.ones(N, dtype=bool)
    
    # Iterative ray tracing (handles recursion without actual recursion)
    for depth in range(max_depth + 1):
        if not np.any(active):
            break
        
        depth_start = time.time()
        active_count = np.sum(active)
        print(f"Depth {depth}: Processing {active_count} active rays...")
        
        # Find intersections for all active rays (using JIT-accelerated version)
        active_idx = np.where(active)[0]
        t_values, surface_indices, normals = find_nearest_intersection_batch_jit(
            current_origins[active_idx],
            current_directions[active_idx],
            surface_data
        )
        
        # Separate hits from misses
        hit_mask_local = surface_indices >= 0
        miss_idx = active_idx[~hit_mask_local]
        hit_idx = active_idx[hit_mask_local]
        
        # Misses get background color (weighted)
        colors[miss_idx] += weights[miss_idx] * background_color
        active[miss_idx] = False
        
        if len(hit_idx) == 0:
            break
        
        # Get hit data
        hit_t = t_values[hit_mask_local]
        hit_surf_idx = surface_indices[hit_mask_local]
        hit_normals = normals[hit_mask_local]
        hit_origins = current_origins[hit_idx]
        hit_directions = current_directions[hit_idx]
        
        # Compute hit points
        hit_points = hit_origins + hit_t[:, np.newaxis] * hit_directions
        
        # Get material indices for hit surfaces (using pre-computed array)
        hit_mat_idx = surface_data['material_indices'][hit_surf_idx]
        
        # View directions (from hit point to camera)
        view_dirs = normalize_batch(hit_origins - hit_points)
        
        # Compute lighting for hit points
        M = len(hit_idx)
        diffuse = np.zeros((M, 3))
        specular = np.zeros((M, 3))
        
        for light in lights:
            light_pos = np.array(light.position, dtype=np.float64)
            light_color = np.array(light.color, dtype=np.float64)
            
            # Light direction for each hit point
            to_light = light_pos - hit_points
            light_dists = np.linalg.norm(to_light, axis=1)
            light_dirs = to_light / light_dists[:, np.newaxis]
            
            # Compute soft shadows (vectorized, JIT-accelerated)
            shadow_hit_mask = np.ones(M, dtype=bool)  # All hits are valid
            
            # Create temporary full-size arrays for shadow computation
            temp_hit_points = np.zeros((M, 3))
            temp_hit_points[:] = hit_points
            temp_mat_idx = np.zeros(M, dtype=np.int32)
            temp_mat_idx[:] = hit_mat_idx
            
            shadow_trans = compute_soft_shadow_batch(
                hit_points + hit_normals * EPSILON,
                shadow_hit_mask,
                light, surface_data, temp_mat_idx, materials_array, n_shadow
            )
            
            # Light intensity with shadow
            light_intensity = ((1 - light.shadow_intensity) + 
                              light.shadow_intensity * shadow_trans)
            
            # Diffuse: material.diffuse * light.color * intensity * max(0, N·L)
            n_dot_l = np.maximum(0, np.sum(hit_normals * light_dirs, axis=1))
            
            mat_diffuse = materials_array['diffuse'][hit_mat_idx]
            diffuse += (mat_diffuse * light_color * 
                       (light_intensity * n_dot_l)[:, np.newaxis])
            
            # Specular: material.specular * light.color * spec_intensity * intensity * max(0, R·V)^shininess
            reflect_dirs = reflect_batch(-light_dirs, hit_normals)
            r_dot_v = np.maximum(0, np.sum(reflect_dirs * view_dirs, axis=1))
            
            mat_specular = materials_array['specular'][hit_mat_idx]
            mat_shininess = materials_array['shininess'][hit_mat_idx]
            
            spec_contrib = (mat_specular * light_color * light.specular_intensity *
                           (light_intensity * np.power(r_dot_v, mat_shininess))[:, np.newaxis])
            specular += spec_contrib
        
        # Get material properties for color computation
        mat_transparency = materials_array['transparency'][hit_mat_idx]
        mat_reflection = materials_array['reflection'][hit_mat_idx]
        
        # Surface color = (diffuse + specular) * (1 - transparency)
        surface_color = (diffuse + specular) * (1 - mat_transparency)[:, np.newaxis]
        
        # Add surface color contribution (weighted)
        colors[hit_idx] += weights[hit_idx] * surface_color
        
        # Prepare for next depth iteration
        # Handle reflection rays
        has_reflection = np.any(mat_reflection > EPSILON, axis=1)
        reflect_idx = hit_idx[has_reflection]
        
        # Handle transparency rays 
        has_transparency = mat_transparency > EPSILON
        
        # Deactivate rays with no reflection AND no transparency
        no_recurse = ~has_reflection & ~has_transparency
        active[hit_idx[no_recurse]] = False
        
        if len(reflect_idx) > 0 and depth < max_depth:
            # Setup reflection rays
            reflect_local_idx = np.where(has_reflection)[0]
            reflect_dirs = reflect_batch(hit_directions[reflect_local_idx], 
                                         hit_normals[reflect_local_idx])
            reflect_origins = hit_points[reflect_local_idx] + hit_normals[reflect_local_idx] * EPSILON
            
            current_origins[reflect_idx] = reflect_origins
            current_directions[reflect_idx] = reflect_dirs
            
            # Update weights for reflection contribution
            weights[reflect_idx] *= mat_reflection[reflect_local_idx]
        else:
            active[reflect_idx] = False
        
        # Handle transparency (rays that continue through surface)
        # Only for rays not already doing reflection
        trans_only = has_transparency & ~has_reflection
        trans_only_idx = hit_idx[trans_only]
        
        if len(trans_only_idx) > 0 and depth < max_depth:
            trans_local_idx = np.where(trans_only)[0]
            # Continue in same direction, offset behind surface
            trans_origins = hit_points[trans_local_idx] - hit_normals[trans_local_idx] * EPSILON
            
            current_origins[trans_only_idx] = trans_origins
            current_directions[trans_only_idx] = hit_directions[trans_local_idx]
            
            # Weight by transparency * background contribution
            weights[trans_only_idx] *= mat_transparency[trans_local_idx, np.newaxis]
            active[trans_only_idx] = True
        
        print(f"  Depth {depth} completed in {time.time() - depth_start:.3f}s")
    
    # Reshape to image
    image = colors.reshape((height, width, 3))
    
    total_time = time.time() - start_time
    print(f"Vectorized rendering complete in {total_time:.1f}s")
    
    return image


def _render_row_chunk(args):
    """
    Worker function to render a chunk of rows.
    Called by multiprocessing pool.
    
    Args:
        args: tuple of (y_start, y_end, camera_data, scene_data)
    
    Returns:
        (y_start, y_end, colors) - the rendered chunk
    """
    y_start, y_end, camera_data, scene_data = args
    
    # Reconstruct camera from serialized data
    from camera import Camera
    camera = Camera(
        camera_data['position'],
        camera_data['look_at'],
        camera_data['up_vector'],
        camera_data['screen_distance'],
        camera_data['screen_width']
    )
    camera.forward = np.array(camera_data['forward'])
    camera.right = np.array(camera_data['right'])
    camera.up = np.array(camera_data['up'])
    camera.screen_height = camera_data['screen_height']
    
    # Extract scene data
    width = scene_data['width']
    height = scene_data['height']
    max_depth = scene_data['max_depth']
    n_shadow = scene_data['n_shadow']
    background_color = np.array(scene_data['background_color'])
    materials_array = scene_data['materials_array']
    surfaces = scene_data['surfaces']
    surface_data = scene_data['surface_data']  # JIT-accelerated intersection data
    lights = scene_data['lights']
    
    # Generate rays for this chunk
    ray_origins, ray_directions = camera.generate_rays_for_rows(width, height, y_start, y_end)
    N = ray_origins.shape[0]
    
    # Initialize colors and weights
    colors = np.zeros((N, 3))
    weights = np.ones((N, 3))
    
    current_origins = ray_origins.copy()
    current_directions = ray_directions.copy()
    active = np.ones(N, dtype=bool)
    
    # Iterative ray tracing
    for depth in range(max_depth + 1):
        if not np.any(active):
            break
        
        active_idx = np.where(active)[0]
        t_values, surface_indices, normals = find_nearest_intersection_batch_jit(
            current_origins[active_idx],
            current_directions[active_idx],
            surface_data
        )
        
        hit_mask_local = surface_indices >= 0
        miss_idx = active_idx[~hit_mask_local]
        hit_idx = active_idx[hit_mask_local]
        
        colors[miss_idx] += weights[miss_idx] * background_color
        active[miss_idx] = False
        
        if len(hit_idx) == 0:
            break
        
        hit_t = t_values[hit_mask_local]
        hit_surf_idx = surface_indices[hit_mask_local]
        hit_normals = normals[hit_mask_local]
        hit_origins = current_origins[hit_idx]
        hit_directions = current_directions[hit_idx]
        
        hit_points = hit_origins + hit_t[:, np.newaxis] * hit_directions
        hit_mat_idx = surface_data['material_indices'][hit_surf_idx]  # Using pre-computed array
        view_dirs = normalize_batch(hit_origins - hit_points)
        
        M = len(hit_idx)
        diffuse = np.zeros((M, 3))
        specular = np.zeros((M, 3))
        
        for light in lights:
            light_pos = np.array(light.position, dtype=np.float64)
            light_color = np.array(light.color, dtype=np.float64)
            
            to_light = light_pos - hit_points
            light_dists = np.linalg.norm(to_light, axis=1)
            light_dirs = to_light / light_dists[:, np.newaxis]
            
            shadow_hit_mask = np.ones(M, dtype=bool)
            temp_mat_idx = np.zeros(M, dtype=np.int32)
            temp_mat_idx[:] = hit_mat_idx
            
            shadow_trans = compute_soft_shadow_batch(
                hit_points + hit_normals * EPSILON,
                shadow_hit_mask,
                light, surface_data, temp_mat_idx, materials_array, n_shadow
            )
            
            light_intensity = ((1 - light.shadow_intensity) + 
                              light.shadow_intensity * shadow_trans)
            
            n_dot_l = np.maximum(0, np.sum(hit_normals * light_dirs, axis=1))
            mat_diffuse = materials_array['diffuse'][hit_mat_idx]
            diffuse += mat_diffuse * light_color * (light_intensity * n_dot_l)[:, np.newaxis]
            
            reflect_dirs = reflect_batch(-light_dirs, hit_normals)
            r_dot_v = np.maximum(0, np.sum(reflect_dirs * view_dirs, axis=1))
            
            mat_specular = materials_array['specular'][hit_mat_idx]
            mat_shininess = materials_array['shininess'][hit_mat_idx]
            
            spec_contrib = (mat_specular * light_color * light.specular_intensity *
                           (light_intensity * np.power(r_dot_v, mat_shininess))[:, np.newaxis])
            specular += spec_contrib
        
        mat_transparency = materials_array['transparency'][hit_mat_idx]
        mat_reflection = materials_array['reflection'][hit_mat_idx]
        
        surface_color = (diffuse + specular) * (1 - mat_transparency)[:, np.newaxis]
        colors[hit_idx] += weights[hit_idx] * surface_color
        
        has_reflection = np.any(mat_reflection > EPSILON, axis=1)
        reflect_idx = hit_idx[has_reflection]
        has_transparency = mat_transparency > EPSILON
        
        no_recurse = ~has_reflection & ~has_transparency
        active[hit_idx[no_recurse]] = False
        
        if len(reflect_idx) > 0 and depth < max_depth:
            reflect_local_idx = np.where(has_reflection)[0]
            reflect_dirs = reflect_batch(hit_directions[reflect_local_idx], 
                                         hit_normals[reflect_local_idx])
            reflect_origins = hit_points[reflect_local_idx] + hit_normals[reflect_local_idx] * EPSILON
            
            current_origins[reflect_idx] = reflect_origins
            current_directions[reflect_idx] = reflect_dirs
            weights[reflect_idx] *= mat_reflection[reflect_local_idx]
        else:
            active[reflect_idx] = False
        
        trans_only = has_transparency & ~has_reflection
        trans_only_idx = hit_idx[trans_only]
        
        if len(trans_only_idx) > 0 and depth < max_depth:
            trans_local_idx = np.where(trans_only)[0]
            trans_origins = hit_points[trans_local_idx] - hit_normals[trans_local_idx] * EPSILON
            
            current_origins[trans_only_idx] = trans_origins
            current_directions[trans_only_idx] = hit_directions[trans_local_idx]
            weights[trans_only_idx] *= mat_transparency[trans_local_idx, np.newaxis]
            active[trans_only_idx] = True
    
    return (y_start, y_end, colors)


def render_parallel(camera, scene_settings, materials, surfaces, lights, width, height, num_workers=None):
    """
    Render the scene using multiprocessing (parallel row-based rendering).
    
    Args:
        num_workers: number of worker processes (default: CPU count)
    """
    import multiprocessing as mp
    import time
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    start_time = time.time()
    
    # Setup camera
    aspect_ratio = width / height
    camera.setup(aspect_ratio)
    print(f"Camera setup complete. Forward: {camera.forward}, Right: {camera.right}, Up: {camera.up}")
    
    max_depth = int(scene_settings.max_recursions)
    n_shadow = int(scene_settings.root_number_shadow_rays)
    background_color = np.array(scene_settings.background_color, dtype=np.float64)
    
    print(f"Max depth: {max_depth}, Shadow rays: {n_shadow}x{n_shadow}={n_shadow*n_shadow} per light")
    print(f"Parallel rendering {width}x{height} with {num_workers} workers...")
    
    # Serialize camera data for workers
    camera_data = {
        'position': camera.position.tolist(),
        'look_at': camera.look_at.tolist(),
        'up_vector': camera.up_vector.tolist(),
        'screen_distance': camera.screen_distance,
        'screen_width': camera.screen_width,
        'forward': camera.forward.tolist(),
        'right': camera.right.tolist(),
        'up': camera.up.tolist(),
        'screen_height': camera.screen_height,
    }
    
    # Pre-compute material properties
    materials_array = {
        'diffuse': np.array([m.diffuse_color for m in materials]),
        'specular': np.array([m.specular_color for m in materials]),
        'reflection': np.array([m.reflection_color for m in materials]),
        'shininess': np.array([m.shininess for m in materials]),
        'transparency': np.array([m.transparency for m in materials]),
    }
    
    # Prepare surface data for JIT-accelerated intersection (done once, shared by workers)
    surface_data = prepare_surface_data(surfaces)
    print(f"Prepared surface data for JIT: {len(surfaces)} surfaces")
    
    # Scene data for workers
    scene_data = {
        'width': width,
        'height': height,
        'max_depth': max_depth,
        'n_shadow': n_shadow,
        'background_color': background_color.tolist(),
        'materials_array': materials_array,
        'surfaces': surfaces,  # Keep for compatibility
        'surface_data': surface_data,  # JIT-accelerated intersection data
        'lights': lights,
    }
    
    # Divide rows into chunks
    rows_per_chunk = max(1, height // (num_workers * 4))  # 4 chunks per worker for load balancing
    chunks = []
    for y_start in range(0, height, rows_per_chunk):
        y_end = min(y_start + rows_per_chunk, height)
        chunks.append((y_start, y_end, camera_data, scene_data))
    
    print(f"Divided into {len(chunks)} chunks of ~{rows_per_chunk} rows each")
    
    # Process chunks in parallel
    pool_start = time.time()
    with mp.Pool(num_workers) as pool:
        results = pool.map(_render_row_chunk, chunks)
    
    print(f"All chunks completed in {time.time() - pool_start:.2f}s")
    
    # Assemble final image
    image = np.zeros((height, width, 3), dtype=np.float64)
    for y_start, y_end, colors in results:
        num_rows = y_end - y_start
        chunk_image = colors.reshape((num_rows, width, 3))
        image[y_start:y_end] = chunk_image
    
    total_time = time.time() - start_time
    print(f"Parallel rendering complete in {total_time:.1f}s")
    
    return image


def save_image(image_array, output_path):
    """Save the rendered image to a file."""
    # Clamp values to [0, 1] then scale to [0, 255]
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image.save(output_path)
    print(f"Image saved to {output_path}")


def main():
    import multiprocessing as mp
    
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--sequential', action='store_true', 
                        help='Use sequential (non-vectorized) renderer')
    parser.add_argument('--vectorized', action='store_true',
                        help='Use single-threaded vectorized renderer')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    
    # Separate objects into materials, surfaces, and lights
    materials, surfaces, lights = separate_objects(objects)
    
    print(f"Scene loaded: {len(materials)} materials, {len(surfaces)} surfaces, {len(lights)} lights")
    print(f"Rendering {args.width}x{args.height} image...")
    
    # Choose renderer (parallel is default)
    if args.sequential:
        print("Using sequential (original) renderer...")
        image_array = render(camera, scene_settings, materials, surfaces, lights, 
                            args.width, args.height)
    elif args.vectorized:
        print("Using vectorized (single-threaded) renderer...")
        image_array = render_vectorized(camera, scene_settings, materials, surfaces, lights, 
                                        args.width, args.height)
    else:
        # Default: parallel rendering
        num_workers = args.workers if args.workers else mp.cpu_count()
        print(f"Using parallel renderer with {num_workers} workers...")
        image_array = render_parallel(camera, scene_settings, materials, surfaces, lights, 
                                      args.width, args.height, num_workers)
    
    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
