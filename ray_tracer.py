import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


EPSILON = 1e-6


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    return v / norm


def normalize_batch(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    return v / norms


def reflect(d, n):
    return d - 2 * np.dot(d, n) * n


def reflect_batch(d, n):
    dot = np.sum(d * n, axis=1, keepdims=True)
    return d - 2 * dot * n


def parse_scene_file(file_path):
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
    N = ray_origins.shape[0]
    
    best_t = np.full(N, max_t if max_t != np.inf else np.inf)
    best_surface_idx = np.full(N, -1, dtype=np.int32)
    best_normals = np.zeros((N, 3))
    
    for surf_idx, surface in enumerate(surfaces):
        t_values, normals = surface.intersect_batch(ray_origins, ray_directions)
        
        closer = t_values < best_t
        best_t[closer] = t_values[closer]
        best_surface_idx[closer] = surf_idx
        best_normals[closer] = normals[closer]
    
    return best_t, best_surface_idx, best_normals


def compute_soft_shadow(hit_point, light, surfaces, materials, n_shadow_rays):
    """Compute soft shadow with transparency. Returns intensity factor (0 to 1)."""
    light_pos = light.position
    
    to_light = light_pos - hit_point
    distance_to_light = np.linalg.norm(to_light)
    light_dir = to_light / distance_to_light
    
    if n_shadow_rays <= 1:
        return compute_shadow_ray_transmission(
            hit_point, light_pos, distance_to_light, surfaces, materials
        )
    
    if abs(light_dir[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    
    right = normalize(np.cross(light_dir, up))
    up = normalize(np.cross(right, light_dir))
    
    cell_size = light.radius / n_shadow_rays
    total_transmission = 0.0
    
    for i in range(n_shadow_rays):
        for j in range(n_shadow_rays):
            offset_i = -light.radius / 2 + cell_size * (i + np.random.random())
            offset_j = -light.radius / 2 + cell_size * (j + np.random.random())
            
            sample_pos = light_pos + right * offset_i + up * offset_j
            to_sample = sample_pos - hit_point
            dist_to_sample = np.linalg.norm(to_sample)
            
            transmission = compute_shadow_ray_transmission(
                hit_point, sample_pos, dist_to_sample, surfaces, materials
            )
            total_transmission += transmission
    
    return total_transmission / (n_shadow_rays * n_shadow_rays)


def compute_soft_shadow_batch(hit_points, hit_mask, light, surfaces, material_indices, materials_array, n_shadow_rays):
    """Compute soft shadows for a batch of hit points."""
    N = hit_points.shape[0]
    transmissions = np.ones(N)
    
    if not np.any(hit_mask):
        return transmissions
    
    light_pos = light.position
    active_indices = np.where(hit_mask)[0]
    active_points = hit_points[active_indices]
    M = len(active_indices)
    
    if n_shadow_rays <= 1:
        to_light = light_pos - active_points
        distances = np.linalg.norm(to_light, axis=1)
        directions = to_light / distances[:, np.newaxis]
        
        shadow_origins = active_points + directions * EPSILON
        
        trans = compute_shadow_transmission_batch(
            shadow_origins, directions, distances - EPSILON, 
            surfaces, material_indices[active_indices], materials_array
        )
        transmissions[active_indices] = trans
        return transmissions
    
    to_light = light_pos - active_points
    light_dists = np.linalg.norm(to_light, axis=1, keepdims=True)
    light_dirs = to_light / light_dists
    
    up_ref = np.zeros((M, 3))
    use_y = np.abs(light_dirs[:, 0]) >= 0.9
    up_ref[~use_y] = [1.0, 0.0, 0.0]
    up_ref[use_y] = [0.0, 1.0, 0.0]
    
    right = np.cross(light_dirs, up_ref)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    right = right / np.maximum(right_norms, EPSILON)
    
    up = np.cross(right, light_dirs)
    up_norms = np.linalg.norm(up, axis=1, keepdims=True)
    up = up / np.maximum(up_norms, EPSILON)
    
    cell_size = light.radius / n_shadow_rays
    total_samples = n_shadow_rays * n_shadow_rays
    total_trans = np.zeros(M)
    
    for i in range(n_shadow_rays):
        for j in range(n_shadow_rays):
            rand_i = np.random.random(M)
            rand_j = np.random.random(M)
            
            offset_i = -light.radius / 2 + cell_size * (i + rand_i)
            offset_j = -light.radius / 2 + cell_size * (j + rand_j)
            
            sample_pos = (light_pos + 
                         offset_i[:, np.newaxis] * right + 
                         offset_j[:, np.newaxis] * up)
            
            to_sample = sample_pos - active_points
            distances = np.linalg.norm(to_sample, axis=1)
            directions = to_sample / distances[:, np.newaxis]
            
            shadow_origins = active_points + directions * EPSILON
            
            trans = compute_shadow_transmission_batch(
                shadow_origins, directions, distances - EPSILON,
                surfaces, material_indices[active_indices], materials_array
            )
            total_trans += trans
    
    transmissions[active_indices] = total_trans / total_samples
    return transmissions


def compute_shadow_transmission_batch(origins, directions, max_distances, surfaces, 
                                      hit_material_indices, materials_array):
    """Compute shadow ray transmission for a batch, accounting for transparency."""
    M = origins.shape[0]
    transmissions = np.ones(M)
    active = np.ones(M, dtype=bool)
    
    current_origins = origins.copy()
    remaining_distances = max_distances.copy()
    
    for _ in range(20):
        if not np.any(active):
            break
        
        active_indices = np.where(active)[0]
        t_vals, surf_idx, _ = find_nearest_intersection_batch(
            current_origins[active_indices],
            directions[active_indices],
            surfaces,
            max_t=np.inf
        )
        
        hit_before_light = (t_vals < remaining_distances[active_indices]) & (surf_idx >= 0)
        
        if not np.any(hit_before_light):
            break
        
        hit_idx = active_indices[hit_before_light]
        blocking_surf_idx = surf_idx[hit_before_light]
        blocking_mat_idx = np.array([surfaces[si].material_index - 1 for si in blocking_surf_idx])
        
        transparencies = materials_array['transparency'][blocking_mat_idx]
        transmissions[hit_idx] *= transparencies
        
        active[hit_idx[transmissions[hit_idx] < EPSILON]] = False
        
        still_active_in_hit = transmissions[hit_idx] >= EPSILON
        if np.any(still_active_in_hit):
            move_idx = hit_idx[still_active_in_hit]
            move_t = t_vals[hit_before_light][still_active_in_hit]
            current_origins[move_idx] += directions[move_idx] * (move_t[:, np.newaxis] + EPSILON)
            remaining_distances[move_idx] -= (move_t + EPSILON)
        
        no_hit_idx = active_indices[~hit_before_light]
        active[no_hit_idx] = False
    
    return transmissions


def compute_shadow_ray_transmission(origin, light_pos, max_distance, surfaces, materials):
    """Compute transmission along a shadow ray, accounting for transparent objects."""
    direction = (light_pos - origin) / max_distance
    current_origin = origin + direction * EPSILON
    remaining_distance = max_distance - EPSILON
    transmission = 1.0
    
    while remaining_distance > EPSILON:
        t, surface, _ = find_nearest_intersection(
            current_origin, direction, surfaces, remaining_distance
        )
        
        if t is None:
            break
        
        material = materials[surface.material_index - 1]
        transmission *= material.transparency
        
        if transmission < EPSILON:
            return 0.0
        
        current_origin = current_origin + direction * (t + EPSILON)
        remaining_distance -= (t + EPSILON)
    
    return transmission


def compute_color(ray_origin, ray_direction, hit_point, normal, surface, 
                  materials, surfaces, lights, scene_settings, depth):
    """Compute the color at a hit point using Phong shading."""
    material = materials[surface.material_index - 1]
    
    background_color = scene_settings.background_color
    n_shadow_rays = int(scene_settings.root_number_shadow_rays)
    
    view_dir = normalize(ray_origin - hit_point)
    
    diffuse = np.zeros(3)
    specular = np.zeros(3)
    
    for light in lights:
        light_pos = light.position
        light_color = light.color
        light_dir = normalize(light_pos - hit_point)
        
        shadow_transmission = compute_soft_shadow(
            hit_point + normal * EPSILON, light, surfaces, materials, n_shadow_rays
        )
        
        light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * shadow_transmission
        
        if light_intensity < EPSILON:
            continue
        
        n_dot_l = max(0.0, np.dot(normal, light_dir))
        diffuse += material.diffuse_color * light_color * light_intensity * n_dot_l
        
        reflect_dir = reflect(-light_dir, normal)
        r_dot_v = max(0.0, np.dot(reflect_dir, view_dir))
        
        if r_dot_v > 0:
            specular += (material.specular_color * light_color * 
                        light.specular_intensity * light_intensity * 
                        (r_dot_v ** material.shininess))
    
    reflection_color = np.zeros(3)
    if depth > 0 and np.any(material.reflection_color > EPSILON):
        reflect_ray_dir = reflect(ray_direction, normal)
        reflect_origin = hit_point + normal * EPSILON
        
        reflected = trace_ray(reflect_origin, reflect_ray_dir, 
                            materials, surfaces, lights, scene_settings, depth - 1)
        reflection_color = material.reflection_color * reflected
    
    bg_color = background_color
    if material.transparency > EPSILON and depth > 0:
        continue_origin = hit_point - normal * EPSILON
        bg_color = trace_ray(continue_origin, ray_direction,
                           materials, surfaces, lights, scene_settings, depth - 1)
    
    output_color = (bg_color * material.transparency + 
                   (diffuse + specular) * (1 - material.transparency) + 
                   reflection_color)
    
    return output_color


def trace_ray(ray_origin, ray_direction, materials, surfaces, lights, scene_settings, depth):
    if depth < 0:
        return np.array(scene_settings.background_color, dtype=np.float64)
    
    t, surface, normal = find_nearest_intersection(ray_origin, ray_direction, surfaces)
    
    if surface is None:
        return np.array(scene_settings.background_color, dtype=np.float64)
    
    hit_point = ray_origin + t * ray_direction
    
    return compute_color(ray_origin, ray_direction, hit_point, normal, surface,
                        materials, surfaces, lights, scene_settings, depth)


def render(camera, scene_settings, materials, surfaces, lights, width, height):
    """Render the scene (sequential version)."""
    import sys
    import time
    
    aspect_ratio = width / height
    camera.setup(aspect_ratio)
    print(f"Camera setup complete. Forward: {camera.forward}, Right: {camera.right}, Up: {camera.up}")
    
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    max_depth = int(scene_settings.max_recursions)
    n_shadow = int(scene_settings.root_number_shadow_rays)
    print(f"Max depth: {max_depth}, Shadow rays: {n_shadow}x{n_shadow}={n_shadow*n_shadow} per light")
    
    start_time = time.time()
    
    for y in range(height):
        row_start = time.time()
        for x in range(width):
            ray_origin, ray_direction = camera.generate_ray(x, y, width, height)
            color = trace_ray(ray_origin, ray_direction, 
                            materials, surfaces, lights, scene_settings, max_depth)
            image[y, x] = color
        
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
    """Render the scene using vectorized operations with stack-based recursion."""
    import time
    
    start_time = time.time()
    
    aspect_ratio = width / height
    camera.setup(aspect_ratio)
    print(f"Camera setup complete. Forward: {camera.forward}, Right: {camera.right}, Up: {camera.up}")
    
    max_depth = int(scene_settings.max_recursions)
    n_shadow = int(scene_settings.root_number_shadow_rays)
    background_color = np.array(scene_settings.background_color, dtype=np.float64)
    
    print(f"Max depth: {max_depth}, Shadow rays: {n_shadow}x{n_shadow}={n_shadow*n_shadow} per light")
    print(f"Vectorized rendering {width}x{height} = {width*height} rays...")
    
    materials_array = {
        'diffuse': np.array([m.diffuse_color for m in materials]),
        'specular': np.array([m.specular_color for m in materials]),
        'reflection': np.array([m.reflection_color for m in materials]),
        'shininess': np.array([m.shininess for m in materials]),
        'transparency': np.array([m.transparency for m in materials]),
    }
    
    ray_gen_start = time.time()
    ray_origins, ray_directions = camera.generate_all_rays(width, height)
    N = ray_origins.shape[0]
    print(f"Generated {N} rays in {time.time() - ray_gen_start:.3f}s")
    
    final_colors = np.zeros((N, 3))
    
    ray_stack = [(
        ray_origins.copy(),
        ray_directions.copy(),
        np.ones((N, 3)),
        np.arange(N),
        0
    )]
    
    total_rays_processed = 0
    
    while ray_stack:
        current_origins, current_directions, weights, pixel_indices, depth = ray_stack.pop()
        
        if depth > max_depth or len(pixel_indices) == 0:
            continue
        
        M = len(pixel_indices)
        total_rays_processed += M
        
        depth_start = time.time()
        print(f"Depth {depth}: Processing {M} rays (stack size: {len(ray_stack)})...")
        
        t_values, surface_indices, normals = find_nearest_intersection_batch(
            current_origins,
            current_directions,
            surfaces
        )
        
        hit_mask = surface_indices >= 0
        miss_mask = ~hit_mask
        
        if np.any(miss_mask):
            miss_pixels = pixel_indices[miss_mask]
            final_colors[miss_pixels] += weights[miss_mask] * background_color
        
        if not np.any(hit_mask):
            print(f"  Depth {depth} completed in {time.time() - depth_start:.3f}s (no hits)")
            continue
        
        hit_idx = np.where(hit_mask)[0]
        hit_t = t_values[hit_mask]
        hit_surf_idx = surface_indices[hit_mask]
        hit_normals = normals[hit_mask]
        hit_origins = current_origins[hit_mask]
        hit_directions = current_directions[hit_mask]
        hit_weights = weights[hit_mask]
        hit_pixels = pixel_indices[hit_mask]
        
        hit_points = hit_origins + hit_t[:, np.newaxis] * hit_directions
        hit_mat_idx = np.array([surfaces[si].material_index - 1 for si in hit_surf_idx])
        view_dirs = normalize_batch(hit_origins - hit_points)
        
        H = len(hit_idx)
        diffuse = np.zeros((H, 3))
        specular = np.zeros((H, 3))
        
        # Compute for each light
        for light in lights:
            light_pos = light.position
            light_color = light.color
            
            to_light = light_pos - hit_points
            light_dists = np.linalg.norm(to_light, axis=1)
            light_dirs = to_light / light_dists[:, np.newaxis]
            
            shadow_hit_mask = np.ones(H, dtype=bool)
            temp_mat_idx = hit_mat_idx.copy()
            
            shadow_trans = compute_soft_shadow_batch(
                hit_points + hit_normals * EPSILON,
                shadow_hit_mask,
                light, surfaces, temp_mat_idx, materials_array, n_shadow
            )
            
            light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * shadow_trans
            
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
        final_colors[hit_pixels] += hit_weights * surface_color
        
        has_transparency = mat_transparency > EPSILON
        if np.any(has_transparency) and depth < max_depth:
            trans_idx = np.where(has_transparency)[0]
            trans_origins = hit_points[trans_idx] - hit_normals[trans_idx] * EPSILON
            trans_directions = hit_directions[trans_idx]
            trans_weights = hit_weights[trans_idx] * mat_transparency[trans_idx, np.newaxis]
            trans_pixels = hit_pixels[trans_idx]
            
            ray_stack.append((
                trans_origins,
                trans_directions,
                trans_weights,
                trans_pixels,
                depth + 1
            ))
        
        has_reflection = np.any(mat_reflection > EPSILON, axis=1)
        if np.any(has_reflection) and depth < max_depth:
            refl_idx = np.where(has_reflection)[0]
            refl_directions = reflect_batch(hit_directions[refl_idx], hit_normals[refl_idx])
            refl_origins = hit_points[refl_idx] + hit_normals[refl_idx] * EPSILON
            refl_weights = hit_weights[refl_idx] * mat_reflection[refl_idx]
            refl_pixels = hit_pixels[refl_idx]
            
            ray_stack.append((
                refl_origins,
                refl_directions,
                refl_weights,
                refl_pixels,
                depth + 1
            ))
        
        print(f"  Depth {depth} completed in {time.time() - depth_start:.3f}s")
    
    image = final_colors.reshape((height, width, 3))
    
    total_time = time.time() - start_time
    print(f"Vectorized rendering complete in {total_time:.1f}s ({total_rays_processed} total rays)")
    
    return image

def save_image(image_array, output_path):
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image.save(output_path)
    print(f"Image saved to {output_path}")


def main():    
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--sequential', action='store_true', 
                        help='Use sequential (non-vectorized) renderer instead of vectorized')  
    args = parser.parse_args()

    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    materials, surfaces, lights = separate_objects(objects)
    
    print(f"Scene loaded: {len(materials)} materials, {len(surfaces)} surfaces, {len(lights)} lights")
    print(f"Rendering {args.width}x{args.height} image...")
    
    if args.sequential:
        print("Using sequential (original) renderer...")
        image_array = render(camera, scene_settings, materials, surfaces, lights, 
                            args.width, args.height)
    else:
        print("Using vectorized  renderer...")
        image_array = render_vectorized(camera, scene_settings, materials, surfaces, lights, 
                                        args.width, args.height)
    
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
