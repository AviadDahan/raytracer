# Usage Guide

## Installation

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI: Rendering a scene

Render a scene file to an image:

```bash
python ray_tracer.py <scene_file> <output_image> [options]
```

Common options:

- **`--width` / `--height`**: output resolution (default `500×500`)
- **`--workers N`**: number of processes for the default parallel renderer (default: CPU count)
- **`--vectorized`**: single-process vectorized renderer
- **`--sequential`**: original sequential pixel loop (slow; useful for debugging)

Examples:

```bash
# Default (parallel)
python ray_tracer.py scenes/pool.txt output/pool.png --width 800 --height 600

# Parallel with explicit worker count
python ray_tracer.py scenes/pool.txt output/pool.png --workers 4

# Single-process vectorized mode
python ray_tracer.py scenes/pool.txt output/pool.png --vectorized

# Sequential mode
python ray_tracer.py scenes/pool.txt output/pool.png --sequential
```

## Scene file format (`.txt`)

Each non-empty line describes one object. Lines beginning with `#` are comments.

### Camera

```
cam px py pz   lx ly lz   ux uy uz   screen_distance screen_width
```

### Scene settings

```
set bgr bgg bgb   root_number_shadow_rays   max_recursions
```

Notes:

- Soft shadows use **N×N** samples per light, where `N = root_number_shadow_rays`.
- `max_recursions` controls recursion depth for reflections/transparency continuation.

### Material

```
mtl dr dg db   sr sg sb   rr rg rb   phong_shininess   transparency
```

- `transparency`: `0` = opaque, `1` = fully transparent
- Materials are referenced by **1-based** index in geometry lines, in the order they appear.

### Geometry

Sphere:

```
sph cx cy cz   radius   material_index
```

Infinite plane:

```
pln nx ny nz   offset   material_index
```

Plane equation is `P · N = offset`.

Cube (axis-aligned):

```
box cx cy cz   scale   material_index
```

Where `scale` is the cube edge length.

### Light

```
lgt px py pz   r g b   specular_intensity   shadow_intensity   radius
```

Notes:

- `radius` controls soft shadow sampling size (area light approximation).
- `shadow_intensity` blends between no shadows (`0`) and full shadows (`1`).

## Programmatic usage

See `docs/API.md` for complete API documentation and examples.

## Troubleshooting

- **Everything is white / washed out**: your lights or material colors may exceed `[0, 1]`. The renderer clamps before saving, which can hide overbright values.
- **Shadow acne / speckles**: increase `EPSILON` slightly in `ray_tracer.py` (trade-off: can cause light leaks).
- **Soft shadows look noisy**: increase `root_number_shadow_rays` (cost grows with \(N^2\)).

