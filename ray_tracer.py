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


# Small epsilon to avoid self-intersection
EPSILON = 1e-6


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    return v / norm


def reflect(d, n):
    """Reflect direction d around normal n."""
    return d - 2 * np.dot(d, n) * n


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
    Render the scene to an image array.
    """
    # Setup camera
    aspect_ratio = width / height
    camera.setup(aspect_ratio)
    
    # Pre-allocate image array
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    max_depth = int(scene_settings.max_recursions)
    
    # Render each pixel
    total_pixels = width * height
    for y in range(height):
        for x in range(width):
            # Generate ray for this pixel
            ray_origin, ray_direction = camera.generate_ray(x, y, width, height)
            
            # Trace the ray
            color = trace_ray(ray_origin, ray_direction, 
                            materials, surfaces, lights, scene_settings, max_depth)
            
            image[y, x] = color
        
        # Progress indicator
        if (y + 1) % 50 == 0 or y == height - 1:
            progress = ((y + 1) * width) / total_pixels * 100
            print(f"Rendering: {progress:.1f}% complete")
    
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
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    
    # Separate objects into materials, surfaces, and lights
    materials, surfaces, lights = separate_objects(objects)
    
    print(f"Scene loaded: {len(materials)} materials, {len(surfaces)} surfaces, {len(lights)} lights")
    print(f"Rendering {args.width}x{args.height} image...")
    
    # Render the scene
    image_array = render(camera, scene_settings, materials, surfaces, lights, 
                        args.width, args.height)
    
    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
