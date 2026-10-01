# Exhibition Template

IARC 425 · *Reading Big Data at the Human Scale* · University of Tennessee, Knoxville

A starting point for the module assignment: a hypothetical exhibition where the
artwork **is** the interior environment. No projections, no sculpture, no sound —
the medium is the temperature, humidity, air speed and light level of the room,
driven by live data.

Two files do the work.

| File | Role |
|---|---|
| `app.py` | The brain. Polls APIs, runs the fluid simulation, applies the if-then rules, serves the site, writes the JSON. |
| `index.html` | The face. Draws the room, the graphs and the diagram. Computes nothing. |

## Run it

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. The first launch spends ~6 s compiling the
Numba kernels; after that it is about 11 ms per frame. Nothing is required from
the network — the four synthetic feeds run offline.

## The two modes

**Place** — drag units around the room, scroll over one to rotate it,
shift-scroll to change its power. Dragging empty space orbits the camera.
When the layout is right, press **Copy placement for Claude** and paste the
result into a chat with the instruction *"hard-code this into app.py"*. The
output is already valid Python for the two edit zones.

**Simulate** — runs the hard-coded layout in `PLACEMENTS` and writes one JSON
file per system into `exports/`.

## Vibecoding

Both files are written to be edited by an AI agent from a plain-English prompt.
Everything a student is likely to change lives inside a marked block:

```
# >>> AI-AGENT EDIT ZONE: RULES >>>
...
# <<< END AI-AGENT EDIT ZONE: RULES <<<
```

```bash
grep -rn "AI-AGENT EDIT ZONE" app.py index.html
```

Nine zones in total — four in `app.py` (API registry, placements, rules,
tuning) and five in `index.html` (theme, room viewer, workflow diagram, field
ramps, statement). Each carries instructions for the agent about what belongs
there and what does not, so "make the fan blow the other way" edits data rather
than the solver.

## Linking an API

Add an entry to `API_REGISTRY` in `app.py`:

```python
{
    "id": "my_feed",
    "label": "Whatever this measures",
    "kind": "http",
    "url": "https://example.org/data.json",
    "path": "results.0.value",     # dot-path; numbers index into lists
    "units": "ppm",
    "range": [0, 500],
    "every": 120,                  # seconds between polls
}
```

## Writing a rule

Rules live in `RULES` in `app.py` and **nowhere else**. The page prints them and
shows which branch is firing, but it cannot edit them and there is no endpoint
that would let it — a student who wants new behaviour has to go and write it.
Add a dict:

```python
{"system": "fan_1", "api": "synthetic_spike", "op": ">", "value": 70,
 "value2": None, "then": 4.5, "otherwise": 0.4},
```

Read as:

> IF `api` `op` `value` THEN setpoint = `then` ELSE setpoint = `otherwise`

Ops are `>` `<` `>=` `<=` `between` `outside`. What a setpoint means depends on
the system: °C for the heat pump, %RH for the humidifier, m/s for the fan,
0–1 for a can light. Restart `python app.py` to pick up changes. A system with
no rule holds its default and says so on the page.

## The room

12 × 8 × 4 m, divided into six zones by the six ceiling can lights:

```
 zone 1 | zone 2 | zone 3      front (y = 0)
 -------+--------+-------
 zone 4 | zone 5 | zone 6      back  (y = D)
```

x runs left to right, y runs front to back, `rot` is the direction a unit
discharges air (0° = +x, 90° = toward the back). The can lights cannot be moved
— they are what defines the zone grid — but they can be dimmed and rule-linked
individually.

## What the simulation actually does

A real 2D incompressible Navier–Stokes solve on a **staggered (MAC) grid**,
96 × 64 cells at 12.5 cm, Numba JIT compiled. Each step:

1. plant output — proportional control toward each system's setpoint
2. in-plane density force + bulk drag
3. semi-Lagrangian advection
4. implicit diffusion (Gauss–Seidel)
5. pressure projection (SOR) — makes the velocity field divergence-free
6. advect and diffuse temperature and humidity
7. envelope exchange with outdoor conditions

Light is not advected. It is solved photometrically from the six ceiling
sources, `E = I·cos θ / d²` — air cannot carry light.

**Measured, not claimed:** 11.4 ms per frame, and RMS divergence of the velocity
field at 0.06 % of the advective scale. The staggered grid is the reason for
that second number: on a MAC grid the discrete divergence and gradient compose
into exactly the Laplacian being solved, so projection actually removes
divergence. The same solve on a collocated grid — Stam's original stable-fluids
arrangement — left 8 % and could not do better at any iteration count.

One honest caveat, also stated in the code: the domain is a **horizontal slice**
through the room at 1.2 m, so vertical buoyancy is out of plane. `BUOY_K` models
the in-plane part of it — the horizontal density gradient that drives a gravity
current. That is a Boussinesq reduction, not a 3D plume. Everything else is
solved as written.

## Grasshopper hand-off

`POST /api/export` (the **Write JSON to exports/** button) writes:

- `<system_id>.json` — one per system: position, rotation, zone, setpoint,
  current output with units, the linked API and whether its condition is met,
  plus the recent time series
- `_room.json` — room dimensions, the six zone centres, and the current mean of
  every field per zone
- `_all_systems.json` — all of the above in one file

In Grasshopper, point a **File Read** component at the file you want and drive
it with a **Timer**. The `position_m` / `rotation_deg` fields place geometry;
`zone_values` in `_room.json` drives whatever you want to vary per zone.

## Deliberate design choices

**System identity is carried by icon and label, never by colour.** Four
categorical hues cannot be reliably told apart under colour-vision deficiency,
and the line-art glyphs already separate them completely. Colour is reserved for
the one thing that needs it — the field gradient.

**Temperature uses a diverging ramp centred on 21 °C**, so you read deviation
from comfort in both directions at once. Given that the assignment's premise is
that perfect interior comfort is no longer the goal, that felt like the right
default. The other three fields are magnitude, so they get sequential
single-hue ramps. All four are declared in OKLCH and converted at load, which
guarantees monotonic lightness and even perceptual spacing.

**The viewer is orthographic, not perspective.** It keeps the drawing an
axonometric like the one in `Diagram Blank.ai`, parallel walls stay parallel so
the room reads as measured — and a horizontal rectangle stays a parallelogram,
so the field is painted onto the slice with one affine transform. The three
surfaces facing the camera are culled at every angle, so you are always looking
into the room.

**No CDN, no build step.** Two files and four pip packages. It runs with the
wifi off.

## Files

```
exhibition-template/
  app.py             backend: solver, APIs, rules, export
  index.html         frontend: viewer, graphs, diagram, statement
  requirements.txt
  exports/           written by simulate mode; read by Grasshopper
```

The artistic statement at the bottom of the page is lorem ipsum. That one is
yours to write.
