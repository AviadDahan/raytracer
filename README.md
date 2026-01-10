# Ray Tracer

A Python Implementation of a ray tracer with Phong shading, soft shadows, reflections, and transparency. 

## Quick Start

```bash
pip install -r requirements.txt
python ray_tracer.py scenes/pool.txt output/pool.png --width 500 --height 500
```

## Project Structure

```
raytracer/
  ray_tracer.py       # Render loops, shading, shadows
  camera.py           # Ray generation
  light.py            # Point light
  material.py         # Material properties
  scene_settings.py   # Background, shadow rays, max depth
  surfaces/
    sphere.py         # Sphere intersection
    infinite_plane.py # Plane intersection
    cube.py           # Cube intersection (slab method)
  scenes/
    pool.txt          # Example scene
  output/             # Rendered images
```

---

## Scene File Format

```
cam   0 1 -3   0 0 0   0 1 0   2 2
set   0.5 0.7 1.0   5   3
mtl   0.8 0.2 0.2   1 1 1   0.3 0.3 0.3   50   0
sph   0 0 0   0.5   1
pln   0 1 0   -0.5   2
box   1 0 0   0.5   3
lgt   2 3 -2   1 1 1   1   0.8   0.5
```

| Line | Format |
|------|--------|
| cam | position, look-at, up, screen_dist, screen_width |
| set | bg_color, shadow_rays, max_recursion |
| mtl | diffuse, specular, reflection, shininess, transparency |
| sph | center, radius, material_idx |
| pln | normal, offset, material_idx |
| box | center, edge_length, material_idx |
| lgt | position, color, spec_intensity, shadow_intensity, radius |

---

## Dependencies

```bash
pip install numpy pillow
```

## Optimizations

The optimized renderer combines vectorization and parallelization for speed.

### The Problem

A naive ray tracer processes one pixel at a time. For a 500x500 image, that's 250,000 sequential operations with Python loop overhead.

### Solution

**1. NumPy Vectorization**

Process all rays in batches instead of one-by-one:

```python
# Sequential: one ray
def intersect(ray_origin, ray_direction):
    oc = ray_origin - self.position
    a = dot(ray_direction, ray_direction)
    b = 2 * dot(oc, ray_direction)
    c = dot(oc, oc) - radius*radius
    ...

# Vectorized: all rays at once
def intersect_batch(ray_origins, ray_directions):
    oc = ray_origins - self.position
    a = np.sum(ray_directions * ray_directions, axis=1)
    b = 2 * np.sum(oc * ray_directions, axis=1)
    c = np.sum(oc * oc, axis=1) - radius*radius
    ...
```

---

## Usage

```bash
python ray_tracer.py <scene_file> <output_image> [options]
```

| Option | Description |
|--------|-------------|
| `--width N` | Image width (default: 500) |
| `--height N` | Image height (default: 500) |
| `--sequential` | Original pixel-by-pixel |

---

## Performance

500x500 image, pool.txt scene:

| Renderer | Time | Speedup |
|----------|------|---------|
| Sequential | ~600s | 1x |
| Vectorized | 14.7s | ~40x |

---

