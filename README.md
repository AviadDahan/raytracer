# Ray Tracer

A Python-based ray tracer implementing the Phong shading model with support for multiple surface types, soft shadows, reflections, and transparency.

## Features

- **Surface Types**: Spheres, Infinite Planes, and Axis-Aligned Cubes
- **Phong Shading**: Diffuse and specular lighting with configurable shininess
- **Soft Shadows**: N×N grid sampling with random jittering for smooth shadow edges
- **Transparency**: Semi-transparent materials with recursive ray continuation
- **Reflections**: Recursive reflection rays with configurable reflection color
- **Transparency in Shadows (Bonus)**: Light passes through transparent objects proportionally

## Changes from Skeleton Code

The original skeleton provided basic data classes with no ray tracing logic. Below is a summary of all additions and modifications:

### New Files

| File | Description |
|------|-------------|
| `requirements.txt` | Dependencies (numpy, pillow) |
| `README.md` | Project documentation |

### Modified Files

#### `ray_tracer.py`
The skeleton only had `parse_scene_file()` and a placeholder `main()`. Added:
- `normalize(v)`, `reflect(d, n)` - Vector utility functions
- `separate_objects(objects)` - Separates parsed objects by type
- `find_nearest_intersection()` - Ray-surface intersection testing
- `compute_soft_shadow()` - N×N grid shadow sampling with jittering
- `compute_shadow_ray_transmission()` - Transparency-aware shadow rays (bonus)
- `compute_color()` - Full Phong shading implementation
- `trace_ray()` - Recursive ray tracing with reflection/transparency
- `render()` - Main render loop with progress reporting
- `save_image()` - Fixed to use output path argument (was hardcoded)

#### `camera.py`
The skeleton only stored constructor parameters. Added:
- Converted all vectors to `np.array` for efficient math
- `setup(aspect_ratio)` - Computes orthonormal camera basis (forward, right, up)
- `generate_ray(x, y, width, height)` - Generates ray through pixel coordinates

#### `surfaces/sphere.py`
The skeleton only stored position, radius, material_index. Added:
- Converted position to `np.array`
- `intersect(ray_origin, ray_direction)` - Quadratic formula intersection, returns (t, normal)

#### `surfaces/infinite_plane.py`
The skeleton only stored normal, offset, material_index. Added:
- Converted normal to `np.array` and normalized it
- `intersect(ray_origin, ray_direction)` - Plane intersection formula, returns (t, normal)

#### `surfaces/cube.py`
The skeleton only stored position, scale, material_index. Added:
- Converted position to `np.array`
- Pre-computed `min_bound` and `max_bound` for efficiency
- `intersect(ray_origin, ray_direction)` - Slab method intersection, returns (t, normal)

#### `light.py`
- Converted position and color to `np.array`

#### `material.py`
- Converted diffuse_color, specular_color, reflection_color to `np.array`

#### `scene_settings.py`
- Converted background_color to `np.array`
- Cast shadow rays and max recursions to `int`

### Skeleton Code Preserved

- Scene file parsing logic in `parse_scene_file()` (unchanged)
- Command-line argument parsing structure (unchanged)
- Class attribute names and constructor signatures (unchanged)

## Usage

```bash
python ray_tracer.py <scene_file> <output_image> [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--width WIDTH` | Image width in pixels (default: 500) |
| `--height HEIGHT` | Image height in pixels (default: 500) |
| `--workers N` | Number of worker processes (default: CPU count) |
| `--vectorized` | Use single-threaded vectorized renderer |
| `--sequential` | Use original sequential renderer |

### Examples

```bash
# Default: parallel rendering with all CPU cores (fastest)
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500

# Parallel with specific worker count
python ray_tracer.py scenes/pool.txt output/pool.png --workers 4

# Single-threaded vectorized rendering
python ray_tracer.py scenes/pool.txt output/pool.png --vectorized
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

---

## Changelog

### v2.0.0 - Performance Optimization Release

This release introduces major performance improvements through NumPy vectorization and multiprocessing parallelization, achieving approximately **200x speedup** over the original sequential implementation.

#### New Rendering Modes

| Mode | Flag | Description |
|------|------|-------------|
| Parallel | *(default)* | Multiprocessing + vectorization (~200x faster) |
| Vectorized | `--vectorized` | NumPy batch processing, single-threaded (~40x faster) |
| Sequential | `--sequential` | Original pixel-by-pixel renderer (baseline) |

#### Usage Examples

```bash
# Parallel rendering (default, fastest, uses all CPU cores)
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500

# Parallel with specific worker count
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500 --workers 8

# Single-threaded vectorized rendering
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500 --vectorized

# Original sequential rendering (for comparison)
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500 --sequential
```

#### Performance Benchmarks

Tested on 500x500 image with pool.txt scene (10 CPU cores):

| Renderer | Time | Speedup |
|----------|------|---------|
| Sequential | ~600s | 1x (baseline) |
| Vectorized | 14.7s | ~40x |
| Parallel (10 workers) | 3.2s | ~200x |

#### Technical Changes

##### NumPy Vectorization

**`camera.py`**
- Added `generate_all_rays(width, height)` - generates all rays as `(H*W, 3)` arrays using `np.meshgrid`
- Added `generate_rays_for_rows(width, height, y_start, y_end)` - generates rays for specific row ranges (for parallel rendering)

**`surfaces/sphere.py`**
- Added `intersect_batch(ray_origins, ray_directions)` - vectorized ray-sphere intersection using batch quadratic formula

**`surfaces/infinite_plane.py`**
- Added `intersect_batch(ray_origins, ray_directions)` - vectorized ray-plane intersection

**`surfaces/cube.py`**
- Added `intersect_batch(ray_origins, ray_directions)` - vectorized slab method for ray-box intersection

**`ray_tracer.py`**
- Added `normalize_batch(v)` - batch vector normalization
- Added `reflect_batch(d, n)` - batch reflection computation
- Added `find_nearest_intersection_batch()` - tests all rays against all surfaces in parallel
- Added `compute_soft_shadow_batch()` - vectorized soft shadow computation with per-point light basis
- Added `compute_shadow_transmission_batch()` - vectorized shadow ray transmission
- Added `render_vectorized()` - main vectorized render loop using iterative depth traversal

##### Multiprocessing Parallelization

**`ray_tracer.py`**
- Added `_render_row_chunk(args)` - worker function that renders a chunk of rows
- Added `render_parallel(camera, scene_settings, materials, surfaces, lights, width, height, num_workers)` - distributes work across CPU cores using `multiprocessing.Pool`
- Row-based chunking with ~4 chunks per worker for load balancing
- Parallel is now the default renderer
- CLI flags: `--vectorized` (single-threaded), `--workers N` (set worker count)

#### Bug Fixes

- Fixed soft shadow light basis computation in vectorized renderer: now correctly computes per-point basis vectors instead of approximating from first point only

#### Architecture

The vectorized renderer transforms the traditional recursive ray tracing into an iterative batch process:

```
Depth 0: Trace all primary rays (250,000 for 500x500)
    └─> Find intersections, compute shading, identify reflection rays
Depth 1: Trace reflection rays (~95,000 active)
    └─> Repeat shading, identify next level reflections
Depth 2: Continue with remaining active rays (~20,000)
    └─> ... continues until max_depth or no active rays
```

The parallel renderer divides the image into row chunks, with each worker running the full vectorized pipeline on its assigned rows independently.

