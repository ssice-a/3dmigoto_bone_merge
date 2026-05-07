# Bone Merge Capture

Blender addon for building a 3Dmigoto bone-capture pipeline around multiple IBs.

The current workflow is manifest-driven. Presets, legacy target scans, and INI repair flows have been removed so the UI matches the runtime design.

## Workflow

1. Set `FrameAnalysis Dir`, then run `Analyze Main`.
2. Review the generated Candidate IB list. Use `+`, `-`, and refresh to adjust it.
3. Import enabled candidates when native meshes are needed in Blender.
4. Run `Build Pool` to create the compact global bone pool and IB region collections.
5. Run `Apply Pool` to rename candidate vertex groups and merge seam groups.
6. Optionally set `LOD FrameAnalysis Dir`, then run `Analyze LOD`.
7. Set `Output Dir`, choose `Buffer Only` or `Buffer + INI`, then run `Export`.

The export root is the single source of truth. Child collections under the export root represent final runtime IB regions, and optional `partNN` children split a region when its actually weighted global vertex groups exceed 256.

## Runtime Shape

The generated runtime does two jobs:

- Capture native current/previous bone matrices into the canonical global pool.
- Gather each exported part's `PartLocalToGlobalBoneMap` into a small local pool before replay, bind that pool as `vs-t0`, and redirect `cb1`.

Main and LOD capture use the same semantic contract. LOD has its own capture map file because its IB hashes and local bone palettes can differ from the main model.

`vb0` is Position, `vb1` is Texcoord, `vb2` is Blend, and runtime binds `vb3` to the same Position resource.

## Safety Notes

- Only exported region IBs are skipped.
- `vs == 200` shadow draws for exported regions are delayed to the final compatible shadow host.
- Visible replay lives under `if vs != 200`.
- LOD replay uses LOD texture overrides but replays the canonical exported geometry and part maps.
- Export blocks if an actually used palette bone is unmatched by LOD until the user applies or accepts LOD fallback repair.

## Git Notes

Do not commit runtime caches:

- `__pycache__/`
- `*.pyc`
- `Buffer/`
- `Meshes/`
- `export_manifest.json`
