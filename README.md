# Bone Merge Capture

Blender addon for building a 3Dmigoto bone-capture pipeline around multiple IBs.

The current workflow is manifest-driven. Presets, legacy target scans, and INI repair flows have been removed so the UI matches the runtime design.

## Workflow

1. Set `FrameAnalysis Dir`, then run `Analyze Main`.
2. Review the generated Candidate IB list. Use `+`, `-`, and refresh to adjust it.
3. Import enabled candidates when native meshes are needed in Blender.
4. Run `Build Pool` to create the compact global bone pool and IB region collections.
5. Run `Apply Pool` to rename candidate vertex groups and merge seam groups.
6. Optionally add one or more LOD Profiles, set each level and FrameAnalysis directory, then run `Analyze Active` for every enabled profile.
7. Set `Output Dir`, choose `Buffer Only` or `Buffer + INI`, then run `Export`.

The export root is the single source of truth. Child collections under the export root represent final runtime IB regions. Direct meshes in a region become implicit `part00` sharing one buffer/palette, but each mesh keeps its own draw range. Optional `partNN` children define explicit exported parts and are required for nested mesh collections.

## Runtime Shape

Bone-merge INI output requires the official `XXMI-Libs` v0.9.4 runtime or
newer. v0.9.2 has FIFO resource pools but cannot parse `HashRegion`.

The generated runtime does two jobs:

- Capture native current/previous bone matrices into the canonical global pool.
- Use EFMI `HashRegion` plus a FIFO CB resource pool to keep same-character instances mapped across shadow and visible passes.
- Gather each exported part's `PartLocalToGlobalBoneMap` into a small local pool before replay, bind that pool as `vs-t0`, and redirect shadow `b1` or visible `b2`.

Main and LOD capture use the same semantic contract. LOD has its own capture map file because its IB hashes and local bone palettes can differ from the main model.

`vb0` is normally Position, `vb1` is Texcoord, and `vb2` is Blend. Runtime aliases `vb3` to `vb0` only when their captured resource identities match; equal layouts alone do not prove an alias.

Vertex import/export follows the physical-field and lossless-carrier contract
in [docs/VERTEX_LAYOUT_ROUNDTRIP.md](docs/VERTEX_LAYOUT_ROUNDTRIP.md).

## Safety Notes

- Only exported region IBs are skipped.
- `vs == 200` shadow draws for exported regions are delayed to the final compatible shadow host.
- Visible replay is allowlisted under `if vs == 201 || vs == 202 || vs == 203`; unknown effect shaders keep their native draw.
- Missing instance mappings and draws beyond the eight-slot runtime capacity keep their native draw.
- CPU pre-skinned imports are named `[CPU_SKINNED_UNSUPPORTED]`; they remain reference-only and do not enter the global bone pool or replacement export.
- Required vertex fields are never silently filled with zero. Imported raw
  carriers preserve unknown formats and padding; external meshes without a
  reconstructible field are rejected with a field-specific export error.
- LOD replay uses LOD texture overrides but replays the canonical exported geometry and part maps.
- Export blocks if an actually used palette bone is unmatched by LOD until the user applies or accepts LOD fallback repair.
- Rebuilding a changed global pool invalidates every analyzed LOD Profile because canonical `Gxx` positions may have moved.
- LOD Mapping rows are diagnostics. LOD Repair checkboxes control which previewed fallback suggestions are applied.
- Two enabled profiles that reuse one override key but map the same canonical bone from different source-local bones are rejected because runtime cannot distinguish them.

## Git Notes

Do not commit runtime caches:

- `__pycache__/`
- `*.pyc`
- `Buffer/`
- `Meshes/`
- `export_manifest.json`
