# Diagrams

Each `.drawio` here is the editable source. The same-named `.png` is what the
docs embed. To change a figure, edit the `.drawio` in draw.io and re-export:

```bash
# adjust: EXPORT_JS points at draw.io's export tool, which is not part of this
# repository, and FIGURE is the file you changed
EXPORT_JS=/path/to/drawio_export/export_drawio.js
FIGURE=pose_graph

node "$EXPORT_JS" "$FIGURE.drawio" "$FIGURE.png" 2
```

The exporter drives draw.io's own renderer headlessly (see the draw.io
`export3.html` tooling). Do not re-render through a third-party converter:
those re-implement the renderer and silently mangle shapes.

`bin_grid_id_interp_setup.png` and `bin_grid_ood_setup.png` have no `.drawio`
source — they are matplotlib output generated from the bin definitions.

Three more have no `.drawio` source either. They are adopted from the study
this repository was cut from, and are reproduced here rather than redrawn:

| Figure | What it is |
|---|---|
| `scene_annotated.png` | Render of the cylindrical room with its parts labelled |
| `bin_grid_in_scene.png` | The bin grid spawned as visual markers, top camera. Cropped from a two-panel figure, so the panel label was trimmed |
| `bin_anatomy.png` | Matplotlib figure of one bin: nine positions, three yaws each |

Pick the export scale by what the figure contains. Label-dense vector figures
need scale 2 — at scale 1 the sub- and superscripts of the transform labels
turn to mush. Figures backed by a screenshot or render are fine at 1.5, which
keeps them from dominating the repository size.

| Figure | Scale |
|---|---|
| `pose_graph`, `pipeline_overview` | 2 |
| `gamepad_mapping`, `cylroom_hemisphere_annotated` | 1.5 |
