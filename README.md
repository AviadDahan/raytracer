# Ray Tracer

A high-performance Python ray tracer implementing Phong shading with soft shadows, reflections, and transparency. Achieves **~200x speedup** over naive implementations through NumPy vectorization and multiprocessing parallelization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Render a scene (uses all CPU cores by default)
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500
```

## Features

- **Surfaces**: Spheres, infinite planes, axis-aligned cubes
- **Phong Shading**: Diffuse and specular lighting
- **Soft Shadows**: N×N jittered grid sampling for smooth shadow edges
- **Reflections**: Recursive reflection rays with configurable tint
- **Transparency**: Light passes through transparent objects proportionally

---

## How It Works: Parallel Vectorized Rendering

The renderer combines two optimization strategies to achieve massive speedups:

### The Problem with Traditional Ray Tracing

A naive ray tracer processes one pixel at a time:

```
for each pixel (x, y):
    generate ray
    trace ray through scene
    compute color
```

For a 500×500 image, this means 250,000 sequential operations—each involving multiple intersection tests, shadow rays, and recursive reflections. This is painfully slow in Python due to interpreter overhead.

### Solution: Vectorization + Parallelization

#### 1. NumPy Vectorization

Instead of processing rays one-by-one, we process them **in batches** using NumPy arrays. This eliminates Python loop overhead and leverages optimized C/Fortran routines.

**Example: Ray-Sphere Intersection**

```python
# Sequential (slow): One ray at a time
def intersect(ray_origin, ray_direction):
    oc = ray_origin - self.position
    a = dot(ray_direction, ray_direction)
    b = 2 * dot(oc, ray_direction)
    c = dot(oc, oc) - radius²
    # ... solve quadratic for single ray

# Vectorized (fast): All rays at once
def intersect_batch(ray_origins, ray_directions):  # (N, 3) arrays
    oc = ray_origins - self.position              # (N, 3)
    a = np.sum(ray_directions * ray_directions, axis=1)  # (N,)
    b = 2 * np.sum(oc * ray_directions, axis=1)          # (N,)
    c = np.sum(oc * oc, axis=1) - radius²                # (N,)
    # ... solve N quadratics in parallel
```

The vectorized version processes 250,000 rays in a single NumPy operation.

#### 2. Iterative Depth Traversal

Traditional ray tracing uses recursion for reflections:

```python
def trace_ray(origin, direction, depth):
    hit = find_intersection(origin, direction)
    color = compute_lighting(hit)
    if has_reflection:
        color += trace_ray(reflect_origin, reflect_dir, depth - 1)  # recursive
    return color
```

Recursion doesn't vectorize. Instead, we use **iterative depth traversal**:

```
Depth 0: Trace ALL 250,000 primary rays at once
         → Find intersections (vectorized)
         → Compute shading (vectorized)
         → Identify rays that need reflection
         
Depth 1: Trace ~95,000 reflection rays at once
         → Repeat shading
         → Identify next level reflections
         
Depth 2: Trace ~20,000 rays
         → Continue until max_depth or no active rays
```

Each depth level is a single batch operation.

#### 3. Multiprocessing Parallelization

Vectorization speeds up single-threaded execution ~40x. To use multiple CPU cores, we add multiprocessing:

```
Image rows divided into chunks:
┌─────────────────────────────┐
│  Chunk 1 (rows 0-30)        │ → Worker 1
├─────────────────────────────┤
│  Chunk 2 (rows 31-60)       │ → Worker 2
├─────────────────────────────┤
│  Chunk 3 (rows 61-90)       │ → Worker 3
├─────────────────────────────┤
│  ...                        │ → ...
└─────────────────────────────┘

Each worker runs the full vectorized pipeline on its chunk independently.
Final image assembled by combining all chunks.
```

The chunks are made small (~4 per worker) for load balancing—some image regions have more reflections/shadows and take longer.

### Soft Shadow Computation

Soft shadows require N×N rays per hit point per light. This is handled in batches:

```python
def compute_soft_shadow_batch(hit_points, light, ...):
    M = len(hit_points)  # e.g., 50,000 hit points
    
    # Compute light plane basis for each point
    to_light = light_pos - hit_points  # (M, 3)
    light_dirs = normalize(to_light)   # (M, 3)
    right = cross(light_dirs, up_ref)  # (M, 3) per-point basis
    up = cross(right, light_dirs)      # (M, 3)
    
    # Sample NxN grid (e.g., 5×5 = 25 shadow rays per point)
    for i in range(N):
        for j in range(N):
            # Random offset within cell (vectorized for all M points)
            rand_i, rand_j = np.random.random(M), np.random.random(M)
            
            # Compute shadow ray directions for all M points at once
            sample_pos = light_pos + offset_i * right + offset_j * up
            directions = normalize(sample_pos - hit_points)
            
            # Trace all M shadow rays simultaneously
            transmission = trace_shadow_batch(hit_points, directions, ...)
            total_trans += transmission
    
    return total_trans / (N * N)
```

For 50,000 hit points with 5×5 shadow sampling and 2 lights, this means **2.5 million shadow rays**—all computed as batch operations.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        render_parallel()                           │
│  • Divides image into row chunks                                   │
│  • Spawns worker processes via multiprocessing.Pool                │
│  • Each worker calls _render_row_chunk()                           │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      _render_row_chunk()                           │
│  • Generates rays for assigned rows (camera.generate_rays_for_rows)│
│  • Runs iterative depth traversal loop                             │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ find_nearest_   │  │ compute_soft_   │  │ reflect_batch() │
│ intersection_   │  │ shadow_batch()  │  │ normalize_batch │
│ batch()         │  │                 │  │                 │
├─────────────────┤  ├─────────────────┤  └─────────────────┘
│ Calls each      │  │ N×N shadow grid │
│ surface's       │  │ with per-point  │
│ intersect_batch │  │ light basis     │
└─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Surface.intersect_batch() methods       │
│ • Sphere: batch quadratic formula       │
│ • Plane: batch ray-plane intersection   │
│ • Cube: batch slab method               │
└─────────────────────────────────────────┘
```

---

## Performance

Benchmarks on 500×500 image with `pool.txt` scene (10 CPU cores):

| Renderer | Time | Speedup |
|----------|------|---------|
| Sequential | ~600s | 1× (baseline) |
| Vectorized | 14.7s | ~40× |
| Parallel (10 workers) | 3.2s | ~200× |

---

## Command Line Options

```bash
python ray_tracer.py <scene_file> <output_image> [options]
```

| Option | Description |
|--------|-------------|
| `--width WIDTH` | Image width in pixels (default: 500) |
| `--height HEIGHT` | Image height in pixels (default: 500) |
| `--workers N` | Number of worker processes (default: all CPUs) |
| `--vectorized` | Single-threaded vectorized renderer |
| `--sequential` | Original pixel-by-pixel renderer |

### Examples

```bash
# Default: parallel rendering (fastest)
python ray_tracer.py scenes/pool.txt output/pool.png

# Use 4 workers
python ray_tracer.py scenes/pool.txt output/pool.png --workers 4

# Single-threaded vectorized
python ray_tracer.py scenes/pool.txt output/pool.png --vectorized

# Original sequential (for comparison)
python ray_tracer.py scenes/pool.txt output/pool.png --sequential
```

---

## Project Structure

```
raytracer/
├── ray_tracer.py          # Main entry: render loops, shading, shadows
├── camera.py              # Ray generation (single + batch)
├── light.py               # Point light definition
├── material.py            # Material properties
├── scene_settings.py      # Background, shadow rays, max depth
├── surfaces/
│   ├── sphere.py          # Sphere intersection (single + batch)
│   ├── infinite_plane.py  # Plane intersection (single + batch)
│   └── cube.py            # Cube intersection (single + batch)
├── scenes/
│   └── pool.txt           # Example scene
└── output/                # Rendered images
```

---

## Scene File Format

```
# Camera: position, look-at, up, screen_distance, screen_width
cam   0 1 -3   0 0 0   0 1 0   2 2

# Settings: background_color, shadow_rays_root, max_recursion
set   0.5 0.7 1.0   5   3

# Material: diffuse, specular, reflection, shininess, transparency
mtl   0.8 0.2 0.2   1 1 1   0.3 0.3 0.3   50   0

# Sphere: center, radius, material_index
sph   0 0 0   0.5   1

# Plane: normal, offset, material_index
pln   0 1 0   -0.5   2

# Box: center, edge_length, material_index
box   1 0 0   0.5   3

# Light: position, color, specular_intensity, shadow_intensity, radius
lgt   2 3 -2   1 1 1   1   0.8   0.5
```

---

## Dependencies

- Python 3.x
- NumPy
- Pillow

```bash
pip install -r requirements.txt
```
