# USD Scene Assets

## Origin

The following USD files are copied from
[lerobot_so101_teleop](https://github.com/liorbenhorin/lerobot_so101_teleop)
by Lior Ben Horin:

- `action-pad-plain.usda` / `action-pad-none.usda` — Flat action pad variants
  (derived from the upstream action pad, logo removed)

The upstream room environment (`room.usda`) and the logo action pad were retired
on 2026-08-13 together with the floor-level table scene; recording, replay and
evaluation all run in the cylindrical room now. They live in git history.

## Original Assets (this workspace)

- `cylindrical_room_shell.usda` — Closed cylindrical room (Ø4m × 3m, interior
  diagonal 5m, no windows) with a single ceiling `DiskLight`, plus a
  rotationally-symmetric pedestal table at the room center. Built procedurally
  via the Isaac Sim MCP executor and exported. Tabletop top surface at world
  `(0, 0, 0.77)` is the SO-ARM101 mount point. A `PhysicsScene` is included.
  Display smoothness via per-prim `refinementLevel=8` (display-only). Not
  derived from upstream — original to this repo.

  The arm is spawned separately by the scene config, so it is intentionally
  absent here. Light prim names (`lights/DiskLight`, `lights/DomeLight`) are
  kept drop-in compatible with the `room_light` / `env_light` prim paths.

  **Collider geometry matters here.** The tabletop carries an explicit
  triangle-mesh collider (`TopCollision`, a real `Mesh` prim). An earlier
  convex-hull tabletop let the solver push the arm *through* the plate: a hull
  resolves penetration towards the nearest face, so past half the plate
  thickness (2.5 cm) the correction flips direction. Three further variants
  existed and were removed: two set `physics:approximation` on `Cylinder`
  *primitives*, which PhysX silently ignores, so despite their names they
  cooked convex hulls anyway. Only a real `Mesh` prim gets a triangle-mesh
  collider. Verify a changed collider by looking at the PhysX collision debug
  visualisation, never by reading the USD attribute.

## License

MIT License, Copyright 2025 Lior Ben Horin.

The original license and copyright notice are included below as required
by the MIT License terms:

> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be included
> in all copies or substantial portions of the Software.

## Known Issues

- The upstream action-pad USDs carry their pivot (Xform origin) at the mesh
  edge, not centered. Yaw rotation via
  `XFormPrim.set_local_poses(orientations=...)` without preserving
  translations clips the pad through the table. Use
  `randomize_static_asset_yaw()` from `lift_cube_env_cfg.py` which
  sets orient via USD API while keeping the translate op intact.
