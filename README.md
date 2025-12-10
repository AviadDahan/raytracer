# Ray Tracer

A Python-based ray tracer implementing the Phong shading model with support for multiple surface types, soft shadows, reflections, and transparency.

## Features

- **Surface Types**: Spheres, Infinite Planes, and Axis-Aligned Cubes
- **Phong Shading**: Diffuse and specular lighting with configurable shininess
- **Soft Shadows**: N×N grid sampling with random jittering for smooth shadow edges
- **Transparency**: Semi-transparent materials with recursive ray continuation
- **Reflections**: Recursive reflection rays with configurable reflection color
- **Transparency in Shadows (Bonus)**: Light passes through transparent objects proportionally

## Usage

```bash
python ray_tracer.py <scene_file> <output_image> [--width WIDTH] [--height HEIGHT]
```

### Example

```bash
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500
```

## Project Structure

```
raytracer/
├── ray_tracer.py          # Main entry point and core ray tracing logic
├── camera.py              # Camera class with ray generation
├── light.py               # Point light source definition
├── material.py            # Material properties (colors, shininess, transparency)
├── scene_settings.py      # Global scene settings
├── surfaces/
│   ├── __init__.py
│   ├── sphere.py          # Sphere surface with ray intersection
│   ├── infinite_plane.py  # Infinite plane surface
│   └── cube.py            # Axis-aligned cube (box) surface
├── scenes/
│   └── pool.txt           # Example scene file
└── output/
    └── *.png              # Rendered images
```

## Component Documentation

### `ray_tracer.py` - Main Ray Tracer

The core module containing:

- **`parse_scene_file(file_path)`**: Parses scene definition files and creates scene objects
- **`separate_objects(objects)`**: Separates parsed objects into materials, surfaces, and lights
- **`find_nearest_intersection(ray_origin, ray_direction, surfaces, max_t)`**: Tests ray against all surfaces, returns nearest hit
- **`compute_soft_shadow(hit_point, light, surfaces, materials, n_shadow_rays)`**: Computes soft shadow with transparency support
- **`compute_shadow_ray_transmission(origin, light_pos, max_distance, surfaces, materials)`**: Traces shadow ray accounting for transparent blockers
- **`compute_color(...)`**: Applies Phong shading model at a hit point
- **`trace_ray(...)`**: Recursively traces a ray through the scene
- **`render(...)`**: Main render loop iterating over all pixels
- **`save_image(image_array, output_path)`**: Saves rendered image with color clamping

**Color Formula** (per assignment specification):
```
output_color = background_color × transparency
             + (diffuse + specular) × (1 - transparency)
             + reflection_color
```

### `camera.py` - Camera

Defines the virtual camera with:

- **Position**: Camera location in 3D space
- **Look-at Point**: Where the camera is aimed
- **Up Vector**: Camera orientation (automatically orthogonalized)
- **Screen Distance**: Focal length controlling viewing angle
- **Screen Width**: Width of the virtual screen

**Methods**:
- **`setup(aspect_ratio)`**: Computes orthonormal basis (forward, right, up vectors)
- **`generate_ray(x, y, width, height)`**: Creates a ray through pixel (x, y)

### `light.py` - Point Light

Defines a point light source with:

- **Position**: Location of the light
- **Color**: RGB light color
- **Specular Intensity**: Multiplier for specular highlights
- **Shadow Intensity**: Controls shadow darkness (0 = no shadow, 1 = full shadow)
- **Radius**: Size of light area for soft shadows

### `material.py` - Material

Defines surface material properties:

- **Diffuse Color**: Base surface color
- **Specular Color**: Color of specular highlights
- **Reflection Color**: Tint applied to reflections
- **Shininess**: Phong exponent controlling highlight sharpness
- **Transparency**: 0 = opaque, 1 = fully transparent

### `scene_settings.py` - Scene Settings

Global scene configuration:

- **Background Color**: Color when rays miss all surfaces
- **Root Number Shadow Rays**: N for N×N shadow sampling grid
- **Max Recursions**: Depth limit for reflection/transparency rays

### `surfaces/sphere.py` - Sphere

Sphere defined by center position and radius.

**Intersection**: Uses quadratic formula to solve ray-sphere intersection.

### `surfaces/infinite_plane.py` - Infinite Plane

Plane defined by normal vector and offset (P·N = offset).

**Intersection**: Direct formula `t = (offset - origin·normal) / (direction·normal)`

### `surfaces/cube.py` - Axis-Aligned Cube

Cube defined by center position and edge length.

**Intersection**: Uses the slab method - tests ray against 6 axis-aligned planes (2 per axis) and determines if entry point is valid.

## Scene File Format

Scene files use a simple text format with one object per line:

```
# Camera: position, look-at, up vector, screen distance, screen width
cam   px py pz   lx ly lz   ux uy uz   sc_dist sc_width

# Settings: background color, shadow rays, max recursion
set   bgr bgg bgb   sh_rays   rec_max

# Material: diffuse, specular, reflection colors, phong coefficient, transparency
mtl   dr dg db   sr sg sb   rr rg rb   phong   trans

# Sphere: center, radius, material index
sph   cx cy cz   radius   mat_idx

# Plane: normal, offset, material index
pln   nx ny nz   offset   mat_idx

# Box: center, edge length, material index
box   cx cy cz   scale   mat_idx

# Light: position, color, specular intensity, shadow intensity, radius
lgt   px py pz   r g b   spec   shadow   width
```

Lines starting with `#` are comments. Material indices are 1-based in order of definition.

## Implementation Details

### Soft Shadows with Transparency (Bonus)

When computing shadows, the ray tracer:

1. Creates an N×N grid on a plane perpendicular to the light direction
2. Samples a random point within each grid cell
3. For each shadow ray:
   - Finds ALL intersecting objects between the point and light
   - Multiplies transmission by each blocker's transparency value
4. Averages transmission across all N² rays
5. Applies: `light_intensity = (1 - shadow_intensity) + shadow_intensity × avg_transmission`

### Performance Optimizations

- All vectors stored as NumPy arrays with `float64` dtype
- Camera basis vectors computed once during setup
- Early exit in intersection tests when `t < 0` or `t > max_dist`
- Pre-allocated image array to avoid repeated allocations
- Materials accessed by index from pre-built list

## Dependencies

- Python 3.x
- NumPy
- Pillow (PIL)

## Example Output

The `pool.txt` scene renders 6 colored spheres on a plane with multiple light sources creating soft shadows.

