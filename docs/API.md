# API Reference

This document describes the **public modules, classes, and functions** in this ray tracer, with practical usage examples.

## Conventions

- **Vectors**: NumPy arrays shaped `(3,)` for single vectors, `(N, 3)` for batches.
- **Colors**: RGB values in **linear** space, typically in the range `[0, 1]`. Output is clamped to `[0, 1]` before saving.
- **Units**: Arbitrary scene units. Keep geometry/light scale consistent.
- **Material indices**: Scene objects store `material_index` as **1-based**; lookups in Python lists/arrays convert to **0-based** (`material_index - 1`).
- **EPSILON**: Small offset used to avoid self-intersection artifacts.

## Module: `ray_tracer.py`

### Constants

- **`EPSILON: float`**  
  Small value (`1e-6`) used for numerical robustness.

### Vector utilities

#### `normalize(v)`

Normalize a single vector.

- **Args**
  - `v`: `(3,)` vector
- **Returns**
  - Normalized vector (or `v` unchanged if its norm is extremely small)

Example:

```python
import numpy as np
from ray_tracer import normalize

v = np.array([3.0, 0.0, 4.0])
print(normalize(v))  # [0.6, 0.0, 0.8]
```

#### `normalize_batch(v)`

Normalize a batch of vectors.

- **Args**
  - `v`: `(N, 3)` vectors
- **Returns**
  - `(N, 3)` normalized vectors

#### `reflect(d, n)`

Reflect direction `d` about surface normal `n`.

- **Args**
  - `d`: `(3,)` direction (typically normalized)
  - `n`: `(3,)` normal (typically normalized)
- **Returns**
  - Reflected direction `(3,)`

#### `reflect_batch(d, n)`

Vectorized reflection for arrays.

- **Args**
  - `d`: `(N, 3)` directions
  - `n`: `(N, 3)` normals
- **Returns**
  - `(N, 3)` reflected directions

### Scene parsing & organization

#### `parse_scene_file(file_path)`

Parse a scene text file and construct the camera, settings, and scene objects.

- **Args**
  - `file_path`: Path to `.txt` scene file
- **Returns**
  - `(camera, scene_settings, objects)`
    - `camera`: `camera.Camera`
    - `scene_settings`: `scene_settings.SceneSettings`
    - `objects`: list containing `Material`, `Light`, and surface objects (`Sphere`, `InfinitePlane`, `Cube`)

Example:

```python
from ray_tracer import parse_scene_file, separate_objects

camera, settings, objects = parse_scene_file("scenes/pool.txt")
materials, surfaces, lights = separate_objects(objects)
```

#### `separate_objects(objects)`

Split the mixed list returned by `parse_scene_file()` into typed lists.

- **Args**
  - `objects`: list
- **Returns**
  - `(materials, surfaces, lights)`

### Ray–scene intersection

#### `find_nearest_intersection(ray_origin, ray_direction, surfaces, max_t=np.inf)`

Test a single ray against all surfaces and return the nearest hit.

- **Args**
  - `ray_origin`: `(3,)`
  - `ray_direction`: `(3,)` (should be normalized)
  - `surfaces`: list of `Sphere | InfinitePlane | Cube`
  - `max_t`: optional distance cutoff
- **Returns**
  - `(t, surface, normal)` if hit, else `(None, None, None)`
    - `t`: distance along ray
    - `surface`: the surface object hit
    - `normal`: `(3,)` surface normal at the hit (oriented per-primitive)

#### `find_nearest_intersection_batch(ray_origins, ray_directions, surfaces, max_t=np.inf)`

Vectorized ray–scene intersection for a batch of rays.

- **Args**
  - `ray_origins`: `(N, 3)`
  - `ray_directions`: `(N, 3)`
  - `surfaces`: list of surfaces supporting `intersect_batch`
  - `max_t`: optional cutoff
- **Returns**
  - `(t_values, surface_indices, normals)`
    - `t_values`: `(N,)` distances (`np.inf` for miss)
    - `surface_indices`: `(N,)` index into `surfaces` (`-1` for miss)
    - `normals`: `(N, 3)` normals for hits (undefined/zero for misses)

### Shadows (hard + soft) with transparency

#### `compute_shadow_ray_transmission(origin, light_pos, max_distance, surfaces, materials)`

Trace a shadow ray and compute **transmission** through potentially transparent blockers.

- **Returns**
  - `transmission` in `[0, 1]` (0 = fully blocked, 1 = fully lit)

Notes:

- Each blocking object multiplies the transmission by its material’s `transparency`.
- The ray origin is offset by `EPSILON` to prevent self-shadowing.

#### `compute_soft_shadow(hit_point, light, surfaces, materials, n_shadow_rays)`

Compute a per-light intensity factor with **N×N** jittered sampling over the light’s area.

- **Args**
  - `n_shadow_rays`: `N` (if `<= 1`, becomes a hard shadow ray)
- **Returns**
  - Average transmission in `[0, 1]`

#### `compute_shadow_transmission_batch(origins, directions, max_distances, surfaces, hit_material_indices, materials_array)`

Vectorized transmission for batches of shadow rays, used by the vectorized renderer.

- **Returns**
  - `(M,)` transmissions

#### `compute_soft_shadow_batch(hit_points, hit_mask, light, surfaces, material_indices, materials_array, n_shadow_rays)`

Vectorized soft shadow computation for a batch of hit points.

- **Returns**
  - `(N,)` transmissions (defaults to `1` for points not in `hit_mask`)

### Shading & ray tracing

#### `compute_color(ray_origin, ray_direction, hit_point, normal, surface, materials, surfaces, lights, scene_settings, depth)`

Compute the shaded color at an intersection using a **Phong** model, with:

- Diffuse + specular from all lights, modulated by soft/hard shadows
- Optional reflections (`material.reflection_color`)
- Optional transparency continuation (`material.transparency`)

The implemented composition is:

\[
color = bg \cdot T + (diffuse + specular)\cdot(1-T) + reflection
\]

Where \(T\) is material transparency.

#### `trace_ray(ray_origin, ray_direction, materials, surfaces, lights, scene_settings, depth)`

Trace a ray and return its color, recursing until `depth` is exhausted.

- **Behavior**
  - Miss → `scene_settings.background_color`
  - Hit → `compute_color(...)`

### Rendering entry points

#### `render(camera, scene_settings, materials, surfaces, lights, width, height)`

Sequential pixel-by-pixel renderer (simplest, slowest).

- **Returns**
  - `image`: `(height, width, 3)` float array in `[0, 1]` (not yet clamped)

#### `render_vectorized(camera, scene_settings, materials, surfaces, lights, width, height)`

Single-process vectorized renderer. Uses batch ray generation + iterative depth traversal.

#### `render_parallel(camera, scene_settings, materials, surfaces, lights, width, height, num_workers=None)`

Multiprocessing renderer. Splits the image into row chunks and runs the vectorized pipeline per chunk.

- **Args**
  - `num_workers`: defaults to CPU count

### Output

#### `save_image(image_array, output_path)`

Clamp the image to `[0, 1]`, convert to `uint8`, and write using Pillow.

Example:

```python
from ray_tracer import save_image
save_image(image_array, "output.png")
```

### CLI

#### `main()`

Command line interface.

Run:

```bash
python ray_tracer.py <scene_file> <output_image> [--width W] [--height H] [--workers N] [--vectorized] [--sequential]
```

Notes:

- Default mode is **parallel** rendering.
- `--sequential` and `--vectorized` are mutually exclusive in practice (the code prioritizes sequential if both are set).

## Module: `camera.py`

### `class Camera`

Represents a pinhole camera with a rectangular image plane.

#### `Camera(position, look_at, up_vector, screen_distance, screen_width)`

- **Args**
  - `position`: `(3,)`
  - `look_at`: `(3,)`
  - `up_vector`: `(3,)` (will be orthogonalized during `setup`)
  - `screen_distance`: float (distance from camera to image plane)
  - `screen_width`: float (physical width of the image plane)

#### `setup(aspect_ratio)`

Compute camera basis vectors `forward`, `right`, `up` and screen height.

- **Args**
  - `aspect_ratio`: `width / height`

#### `generate_ray(x, y, image_width, image_height)`

Generate a single primary ray for pixel `(x, y)` (top-left origin).

- **Returns**
  - `(origin, direction)` where `origin` is the camera position and `direction` is normalized.

#### `generate_all_rays(image_width, image_height)`

Vectorized generation for all pixels.

- **Returns**
  - `origins`: `(H*W, 3)` (tiled camera position)
  - `directions`: `(H*W, 3)` normalized

#### `generate_rays_for_rows(image_width, image_height, y_start, y_end)`

Vectorized generation for a range of rows, used by `render_parallel()`.

## Module: `light.py`

### `class Light`

Point/area light representation (area is simulated via radius sampling).

#### `Light(position, color, specular_intensity, shadow_intensity, radius)`

- **Fields**
  - `position`: `(3,)`
  - `color`: `(3,)` RGB
  - `specular_intensity`: float multiplier on specular term
  - `shadow_intensity`: float in `[0, 1]` (0 = no shadows, 1 = full shadows)
  - `radius`: float controlling area-light sampling size

## Module: `material.py`

### `class Material`

Phong material plus reflection and transparency.

#### `Material(diffuse_color, specular_color, reflection_color, shininess, transparency)`

- **Fields**
  - `diffuse_color`: `(3,)`
  - `specular_color`: `(3,)`
  - `reflection_color`: `(3,)` (tint on reflected ray color)
  - `shininess`: float (Phong exponent)
  - `transparency`: float in `[0, 1]` (0 = opaque, 1 = fully transparent)

## Module: `scene_settings.py`

### `class SceneSettings`

Global scene configuration.

#### `SceneSettings(background_color, root_number_shadow_rays, max_recursions)`

- **Fields**
  - `background_color`: `(3,)`
  - `root_number_shadow_rays`: int `N` used for `N×N` soft-shadow sampling
  - `max_recursions`: int maximum recursion depth

## Package: `surfaces/`

All surfaces share:

- `material_index` (1-based)
- `intersect(ray_origin, ray_direction)` → `(t, normal)` or `(None, None)`
- `intersect_batch(ray_origins, ray_directions)` → `(t_values, normals)`

### Module: `surfaces/sphere.py`

#### `class Sphere(position, radius, material_index)`

Sphere centered at `position` with given `radius`.

- **`intersect(...)`**: quadratic ray–sphere intersection, returns smallest positive `t`.
- **`intersect_batch(...)`**: vectorized quadratic intersection, returns `np.inf` for misses.

### Module: `surfaces/infinite_plane.py`

#### `class InfinitePlane(normal, offset, material_index)`

Plane defined by:

- normalized normal vector `N`
- offset `d` such that points satisfy `P · N = d`

Normals returned are oriented to face the ray origin (i.e., flipped if needed).

### Module: `surfaces/cube.py`

#### `class Cube(position, scale, material_index)`

Axis-aligned cube centered at `position` with edge length `scale`.

- **`intersect(...)`**: slab method, returns entry hit if outside, exit hit if inside.
- **`intersect_batch(...)`**: vectorized slab method, but still uses small Python loops for per-ray normal selection on some paths.

## Programmatic usage examples

### Render from a scene file

```python
from ray_tracer import parse_scene_file, separate_objects, render_parallel, save_image

camera, settings, objects = parse_scene_file("scenes/pool.txt")
materials, surfaces, lights = separate_objects(objects)

img = render_parallel(camera, settings, materials, surfaces, lights, width=500, height=500, num_workers=4)
save_image(img, "output/pool.png")
```

### Build a minimal scene in code (no scene file)

```python
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.sphere import Sphere
from surfaces.infinite_plane import InfinitePlane
from ray_tracer import render_vectorized, save_image

camera = Camera(
    position=[0, 1, -5],
    look_at=[0, 1, 0],
    up_vector=[0, 1, 0],
    screen_distance=1.0,
    screen_width=2.0,
)

settings = SceneSettings(background_color=[0.05, 0.05, 0.08], root_number_shadow_rays=3, max_recursions=3)

materials = [
    Material(diffuse_color=[0.8, 0.2, 0.2], specular_color=[1, 1, 1], reflection_color=[0.2, 0.2, 0.2], shininess=64, transparency=0.0),
    Material(diffuse_color=[0.2, 0.2, 0.2], specular_color=[0.2, 0.2, 0.2], reflection_color=[0, 0, 0], shininess=8, transparency=0.0),
]

surfaces = [
    Sphere(position=[0, 1, 0], radius=1.0, material_index=1),
    InfinitePlane(normal=[0, 1, 0], offset=0.0, material_index=2),
]

lights = [
    Light(position=[3, 6, -3], color=[1, 1, 1], specular_intensity=1.0, shadow_intensity=0.8, radius=1.0),
]

img = render_vectorized(camera, settings, materials, surfaces, lights, width=400, height=300)
save_image(img, "output/minimal.png")
```

