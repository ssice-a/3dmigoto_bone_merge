# Bone Merge Capture Design

## Core Goal

Bone Merge Capture exists to let a final draw use a bone palette that is larger or different from the palette originally attached to that draw.

The runtime model is split into two phases:

1. Capture native bone palettes from game draws before the final merged draw needs them.
2. Rebuild the final draw's local bone window from the captured global store, then redirect the draw's original VS bindings to that rebuilt window.

The plugin should optimize for runtime performance first. Reuse and cleanliness matter, but no abstraction should add extra draw-time CS work without a clear need.

## Render Facts Observed

The game stores many character parts in shared large IB/VB buffers. FrameAnalysis dump filenames expose per-part IB hashes even when the bound `IASetIndexBuffer` resource hash is only the shared backing buffer.

For the tested `LX.ini` character:

- Multiple passes can use the same part IB and `match_index_count`.
- `vs-t0` was stable across tested passes: `554904b3`.
- `VSSetConstantBuffers1(StartSlot:1)` can use different `first_constant` values per pass.
- The logical `cb1` content can differ per pass.
- The bone-window pointer fields used by current capture logic, `cb1[5].x` and `cb1[5].y`, remained stable in tested normal and afterimage passes.
- `cb1[5].z` differed between normal and afterimage/effect passes, for example `258`, `2`, and `1`.

Therefore the design must not assume that the whole `cb1` is identical for all passes of one IB. The runtime should preserve the current draw's `cb1[5].z/w` and only rewrite `cb1[5].x/y` when redirecting to the local rebuilt palette.

## Runtime Identity

The minimal identity for a native bone palette capture is:

```text
part_ib_hash
match_index_count
vs_t0_hash
cb1[5].x
cb1[5].y
local_bone_count
```

Shader hash and pass identity are useful for classification and diagnostics, but they should not be the primary bone-store identity. Shader hashes are expected to change more often during game updates than the underlying mesh layout and bone palette shape.

`cb1[5].x/y` are offline validation fields in the default architecture. They are not read every frame by the runtime shaders when they match the expected local layout.

The final consume identity is:

```text
final_chunk_ib_hash
match_index_count
chunk_index
local_palette
```

`local_palette` maps final local bone index to capture-store bone index.

First implementation scope: multi-instance characters are not supported. The analyzer assumes the selected character appears once in the relevant FrameAnalysis capture, and the runtime plan assumes the matching IB/draw sequence belongs to that one character instance. If the same character/source identity appears multiple times in one frame with overlapping draw sequences, Analyze should stop with a `multi_instance_unsupported` diagnostic instead of generating tables that may capture matrices from the wrong instance.

## Stable Global Bone Construction

Hash-only placement is not stable enough across game updates. The global store should be generated from verified capture records, not from a hardcoded hash order.

At runtime, capture is already mounted inside a `TextureOverride`. That means the runtime trigger is naturally filtered by the current IB hash and `match_index_count`. The compute shader does not need to rediscover the part identity. It should receive a compact meta record whose meaning is generated and validated offline.

A capture meta record should keep the HLSL semantics explicit:

```text
x = capture_store_base
y = source_local_bone_count
z = reserved
w = reserved
```

`source_index_count` stays in the offline manifest as a validation and ordering field. It must not be passed to runtime shaders unless a future shader genuinely needs it.

Do not generate one INI `ResourceBoneMeta_*` section per IB. Per-IB INI resources make the runtime config noisy and were a failed older design. Capture metadata should be materialized into one static buffer file, for example:

```text
Buffer/CaptureMeta.buf
```

The static record format should be:

```text
uint4 CaptureMeta[N]

x = capture_store_base
y = source_local_bone_count
z = reserved
w = reserved
```

At runtime, the `TextureOverride` hash and `if vs == ...` filter already select the capture draw. The INI does not need to pass a dynamic record index into HLSL. The generated INI should bind the matching capture meta record for that override directly, using the simplest 3Dmigoto resource binding available for a one-record view/slice of the unified static buffer.

A capture record should be accepted only after matching several independent signals:

- Part identity: dumped part IB hash and `match_index_count`.
- Mesh shape: index count, vertex count, vertex stride/layout, and optionally index/VB content digest.
- Bone shape: local bone count inferred from Blender groups or extracted blend indices.
- Runtime palette window: `vs_t0_hash` plus `cb1[5].x/y`.
- Draw order: first valid capture draw before the final host draw.

The global store placement should be deterministic after validation:

1. Build candidate capture records from FrameAnalysis.
2. Group records by native palette identity.
3. Validate each group against the expected target mesh/bone shape.
4. Sort accepted groups by a stable manifest order, not by raw hash alone.
5. Assign `capture_store_base` sequentially.
6. Write the resolved mapping into `capture_manifest.json`.

The stable manifest order is based on the user-confirmed Candidate IB list's `match_index_count`, sorted from large to small. The scanner must not depend on fixed VS hashes. Shader hashes are frame-local evidence only.

Analyze should use the g-buffer/display stage as the strongest anchor for locating the target character and then walk back to the early shadow/capture stage to discover the two current shadow VS hashes. This makes the scan resilient when game updates change shader hashes: as long as the visible mesh IBs can be found in g-buffer/display draws and matched back to early shadow draws, the shadow VS pair can be rediscovered.

Transparent parts may not have a g-buffer pass, so g-buffer is not the only source of import candidates. Import candidates can still be discovered from the whole frame as unique IB/VB geometry. The g-buffer anchor is used to resolve the character and the shadow VS pair; the editable Candidate IB list remains the final user-confirmed source for import and global-pool building.

The intended rule is:

1. Analyze FrameAnalysis and populate the editable Candidate IB list.
2. Let the user remove unneeded candidates or add/refresh candidates from scene objects.
3. For each enabled candidate, verify whether the same IB slice appears in the discovered early shadow/capture stage.
4. Exclude enabled candidates that cannot provide an early-shadow palette from global-pool construction.
5. Sort accepted capture records by descending `match_index_count`.
6. Assign `capture_store_base` sequentially in that sorted order.

This keeps large, high-bone parts earlier in the global table and avoids making the global bone layout depend on hash values that may change after a game update.

If two accepted records have the same `index_count`, use deterministic tie-breakers:

```text
descending local_bone_count
ascending match_index_count
ascending mesh_fingerprint
ascending part_ib_hash
```

This makes the global bone positions reproducible while still allowing the scanner to detect when a game update changed a hash. Hash is only a final tie-breaker after stronger structural ordering features.

## Update Resilience

When a game update changes hashes, the scanner should try to relink old targets by structure:

- Prefer exact match on `match_index_count`, vertex count, local bone count, and mesh digest.
- If exact digest is unavailable, use index count, VB layout, vertex count, and a sampled position/blend fingerprint.
- Treat shader hash and draw index as weak hints only.
- If multiple candidates remain, require manual confirmation instead of silently choosing.

The plugin should report relink confidence in the manifest:

```text
exact_hash
structural_exact
structural_probable
ambiguous
missing
```

Only `exact_hash` and `structural_exact` should be allowed in performance-first automatic export. Probable or ambiguous matches should stop generation unless the user explicitly accepts them.

## Manifest Schema

The preset system is removed. A manifest is the only source of truth for one scan/export build. It is not a long-lived reusable preset; it is regenerated from the current FrameAnalysis folder and current export collection.

Top-level shape:

```json
{
  "version": 2,
  "frameanalysis": {},
  "capture_records": [],
  "buffer_tables": {},
  "shadow_skip_records": [],
  "shadow_replay_plan": {},
  "validation": {}
}
```

### frameanalysis

```json
{
  "path": "E:/XXMI/EFMI/FrameAnalysis-2026-05-05-225007",
  "log_path": "E:/XXMI/EFMI/FrameAnalysis-2026-05-05-225007/log.txt",
  "scanned_at": "2026-05-06T00:00:00+08:00"
}
```

### capture_records

Each record represents one accepted native source palette. Records are sorted by descending `source_index_count` before `capture_store_base` is assigned.

```json
{
  "source_ib_hash": "2e5d9294",
  "match_index_count": 23220,
  "source_index_count": 23220,
  "source_local_bone_count": 83,
  "capture_store_base": 0,
  "capture_meta_index": 0,
  "vs_t0_hash": "554904b3",
  "cb1_xy": [0, 1024],
  "capture_vs_filter": 200,
  "capture_draw_indices": [211],
  "confidence": "exact_hash",
  "mesh_fingerprint": ""
}
```

Runtime uses only:

```text
capture_store_base
source_local_bone_count
```

The other fields are offline validation, sorting, diagnostics, and INI matching data.

### buffer_tables

```json
{
  "global_bone_count": 420,
  "max_local_bone_count": 256,
  "global_previous_row_offset": 100000,
  "local_previous_row_offset": 1024,
  "capture_meta_path": "Buffer/CaptureMeta.buf",
  "palette_meta_path": "Buffer/PaletteMeta.buf",
  "palette_table_path": "Buffer/PaletteTable.buf",
  "global_bone_store_rows": 101263,
  "local_bone_store_rows": 1795
}
```

The generator calculates store sizes from the manifest:

```text
global_bone_store_rows = global_previous_row_offset + 3 + global_bone_count * 3
local_bone_store_rows  = local_previous_row_offset  + 3 + max_local_bone_count * 3
```

### shadow_skip_records

Only modified/exported source IBs are skipped at their original shadow draw. Unmodified game parts are left untouched.

```json
{
  "source_ib_hash": "2e5d9294",
  "match_index_count": 23220,
  "shadow_vs_filter": 202,
  "skip_original": true,
  "reason": "exported_shadow_replayed"
}
```

### shadow_replay_plan

The replay host is a late compatible shadow draw used to replay all delayed modified shadow chunks after the global bone store is complete.

```json
{
  "host_ib_hash": "cb71c5cd",
  "host_match_index_count": 21849,
  "host_vs_filter": 202,
  "host_draw_index": 420,
  "batches": [
    {
      "name": "transparent_shadow",
      "set_white_ps_t0": false,
      "chunks": []
    },
    {
      "name": "normal_shadow",
      "set_white_ps_t0": true,
      "chunks": []
    }
  ]
}
```

Each chunk entry should include the exported mesh resources, draw calls, and palette record:

```json
{
  "source_ib_hash": "2e5d9294",
  "match_index_count": 23220,
  "chunk_index": 0,
  "palette_meta_index": 0,
  "mesh_resources": {
    "ib": "Resource_2e5d9294_23220_0_Index",
    "vb0": "Resource_2e5d9294_23220_0_Position",
    "vb1": "Resource_2e5d9294_23220_0_Texcoord",
    "vb2": "Resource_2e5d9294_23220_0_Blend",
    "vb3": "Resource_2e5d9294_23220_0_Position"
  },
  "draws": [
    {
      "index_count": 37008,
      "start_index": 0,
      "base_vertex": 0
    }
  ]
}
```

### validation

```json
{
  "cb1_xy_fast_path": true,
  "all_required_captures_before_replay_host": true,
  "max_local_bone_count_ok": true,
  "ambiguous_records": [],
  "missing_records": [],
  "warnings": []
}
```

INI and buffer generation must stop when required validation fails.

## Runtime Pipeline

Capture pass:

```ini
if vs == <capture_filter>
  cs-t2 = ResourceCaptureMeta_<record_view>
  run = CustomShader_RecordBones
endif
```

`RecordBones` reads the current draw's `vs-t0` by reference. In D3D11, a shader resource slot binds an SRV view, not just a raw underlying buffer. If the game created the SRV as a slice of a larger bone buffer, HLSL indexing is relative to that SRV view. Therefore `vs_t0[0]` means element zero of the currently bound view.

Because scanned target draws currently validate to:

```text
cb1[5].x = 0
cb1[5].y = 1024
```

the default capture path does not need to extract or read `cb1` at runtime. It copies from the fixed view-relative layout:

```text
current:  0    + 3 + localBone * 3 + row
previous: 1024 + 3 + localBone * 3 + row
```

into the global capture store at `capture_store_base`.

If a future scan finds a target whose `cb1[5].x/y` do not match `0/1024`, that target is not compatible with the fast path. It should be reported as needing a fallback runtime mode rather than silently generating incorrect capture code.

Capture-only overrides may suppress original drawing only for source draws that have exported replacement geometry in the current export collection. Skipping an original draw without issuing replacement draw calls will create missing model parts. Therefore draw suppression must live inside the guarded branch that also emits the replacement path for that exported source.

The global store layout must always match this shader meaning:

```text
global current row  = 3 + capture_store_index * 3 + row
global previous row = GLOBAL_PREVIOUS_ROW_OFFSET + 3 + capture_store_index * 3 + row
capture_store_index = capture_store_base + localBone
```

`capture_store_base` is measured in bones, not rows. `source_local_bone_count` is also measured in bones. Any frontend, manifest, or generator code that treats these values as rows is wrong.

Consume pass:

```ini
cs-t2 = ResourceLocalPalette_<chunk>
cs-t3 = ResourceLocalPaletteMeta_<chunk>
run = CustomShader_GatherBones
vs-t0 = ResourceLocalFakeT0_SRV
```

This is the preferred fast path when the original draw already has:

```text
cb1[5].x = 0
cb1[5].y = LOCAL_PREVIOUS_ROW_OFFSET
```

In that case the VS already points at the same row layout used by `ResourceLocalFakeT0_SRV`, so the consume phase only needs to gather the required bones into the local fake `vs-t0` and bind that resource. It does not need to copy or rewrite `cb1`.

Fallback consume pass:

```ini
run = CustomShader_ExtractCB1
cs-t2 = ResourceLocalPalette_<chunk>
cs-t3 = ResourceLocalPaletteMeta_<chunk>
run = CustomShader_GatherBones
vs-t0 = ResourceLocalFakeT0_SRV
run = CustomShader_RedirectCB1
vs-cb1 = ResourceFakeCB1
```

The fallback is only needed when the original draw's `cb1[5].x/y` do not match the local fake `vs-t0` layout. `RedirectCB1` must preserve all current `cb1` values except `cb1[5].x/y`, which are rewritten to the local fake palette bases.

## Performance Direction

Avoid per-pass duplicate capture. If multiple passes share the same native palette identity, capture once where possible and let later consume passes reuse the same global store rows.

The scanner should choose a capture draw that appears before all consuming draws that need it. If that cannot be guaranteed, the integration layer should move the consuming draw to a later host or split the affected chunk.

Future optimization should target reducing draw-time CS count:

- Deduplicate capture records by native palette identity.
- Skip consume work on untouched original draws.
- Keep local palettes as static buffers generated offline.
- Keep capture metadata in a single static buffer generated offline.
- Avoid runtime structural checks; all verification should happen during scan/export.

## Static Buffer Tables

Runtime data should be table-driven and generated offline.

### CaptureMeta

```text
Buffer/CaptureMeta.buf
uint4 CaptureMeta[N]

x = capture_store_base
y = source_local_bone_count
z = reserved
w = reserved
```

Each capture `TextureOverride` binds the corresponding one-record view/slice as `cs-t2`. The shader reads only record zero from that bound view.

### PaletteMeta

```text
Buffer/PaletteMeta.buf
uint4 PaletteMeta[M]

x = palette_table_base
y = local_bone_count
z = reserved
w = reserved
```

### PaletteTable

```text
Buffer/PaletteTable.buf
uint PaletteTable[K]

PaletteTable[palette_table_base + localBone] = capture_store_index
```

Each consume `TextureOverride` binds the corresponding one-record `PaletteMeta` view/slice plus the unified `PaletteTable`. `GatherBones` uses those two buffers to copy from the global capture store into the local fake `vs-t0`.

## Blender Interaction Layer

The Blender UI should stay minimal. The addon should expose the build flow, not the internal runtime mechanics.

This project should become a complete mod-making plugin rather than a standalone auxiliary BoneStore generator. Import, export, texture marking, and base INI generation should reuse the implementation and rules from `E:/vscode/mod_importer` wherever possible.

Reuse targets:

```text
FrameAnalysis discovery / Analyze
model import
texture candidate discovery and marking
collection-driven export
buffer writing
INI generation rules
NTMI fast-path resource naming where applicable
```

Texture marking UI and candidate discovery should be copied from `mod_importer` rather than redesigned. The generated INI replacement style is different: BoneMerge replaces marked textures by texture hash `TextureOverride`, not by writing per-region `ps-t` slot bindings.

Inherited texture workflow:

```text
Analyze gathers PS texture candidates from visible/g-buffer-like draws.
The UI shows region + draw + ps-t slot candidates.
Users mark candidates as base_color, normal, material, or effect.
base_color and normal are unique per region.
material and effect may have multiple indexed marks.
Apply Texture Marks updates Blender materials on already imported/export objects.
Export copies or converts marked textures and writes texture-hash replacement overrides.
```

Inherited storage:

```text
collection custom property: modimp_texture_marks_text
text datablock JSON: candidates / marks / default_draws
region collection property: modimp_texture_slots
```

Inherited INI behavior:

```text
If a region has valid marked textures, generated INI emits texture hash overrides.
If no valid mark exists for a region, generated INI does not invent texture replacements.
DDS sources are copied directly when possible.
Non-DDS sources are converted through the mod_importer texconv path.
```

Texture replacement contract:

```ini
[TextureOverride_<semantic>_<texture_hash>]
hash = <source_texture_hash>
this = Resource_<replacement_texture>
```

The draw's `ps-t` slot remains useful analysis metadata because it helps identify base color, normal, material, and effect candidates. It is not the final replacement key. The final key is the sampled texture resource hash, so the same source texture can be replaced consistently wherever the game samples it.

Because replacement is hash-scoped, one source texture hash should resolve to one replacement resource. If two material marks try to replace the same source hash with different resources, export should stop with a conflict diagnostic unless the user resolves the mark.

Recommended workflow:

```text
1. Select Main FrameAnalysis folder, optional LOD FrameAnalysis folders, and target IB
2. Analyze candidate IBs
3. Edit the candidate IB list
4. Import selected IB models
5. Add source objects to the global-pool UI list
6. Build Global Bone Pool
7. Analyze LOD mappings, if LOD FrameAnalysis folders are configured
8. Organize/edit meshes inside the same single root collection
9. Generate buffers, BoneMerge runtime tables, and INI
10. Inject/modify main INI, if enabled
```

Remove user-facing preset actions. A manifest is regenerated for each build and should not be presented as a reusable mapping preset.

The main panel should show only:

```text
Main FrameAnalysis path
LOD FrameAnalysis paths, optional
Export collection
Output folder
Scan button
Generate button
Inject button
Status summary
```

Status summary should be compact:

```text
capture records: N
global bones: N
modified shadow skips: N
replay chunks: N
validation: pass/fail
```

Detailed diagnostics belong in the manifest and a log panel/file, not in the main UI.

Open UI questions:

```text
How should collections declare transparent_shadow vs normal_shadow?
Should the export collection structure encode replay batch names?
Should ambiguous structural relinks stop with a modal report or a side-panel warning?
Should Inject modify the main INI directly or generate a separate include block first?
How should LOD variants be analyzed and exported automatically?
```

LOD automation is an optional phase after the main analyze/import/global-pool path is stable for the current project. It must not block users who only want the main model path, but once configured it should generate runtime capture and replay overrides the same way the main path does.

## Export Collection Contract

There is exactly one user-facing root collection. Import and export both use this same collection. Do not create separate import and export roots.

This root collection is a large working collection containing imported source objects and export hash subcollections. Export recursively walks mesh objects under valid hash subcollections in this root and writes buffers/INI from that structure.

Rules:

```text
ExportRoot
  <ib_hash>-<match_index_count>
    mesh objects...
  <ib_hash>-<match_index_count>-<chunk_index>
    mesh objects...
```

The hash collection controls the output IB identity and match range. Mesh object names are free-form during export. Object names must not be used as the source bone mapping identity.

### Hash Collection Identity

Each export hash collection name must clearly contain the output IB hash and `match_index_count`. The recommended names are:

```text
<ib_hash>-<match_index_count>
<ib_hash>-<match_index_count>-<chunk_index>
```

Examples:

```text
2e5d9294-23220
2e5d9294-23220-0
2e5d9294-23220-1
```

`match_index_count` is intentionally visible in the collection name so the exported draw identity is clear in Blender. The scanner still verifies it against the current FrameAnalysis.

### Bone Localization

Before export, the addon may clean empty vertex groups. Then it builds the final chunk palette from the vertex groups already present on objects inside each hash collection.

The export-time assumption follows `mod_importer`:

```text
Blender vertex groups use global bone-number semantics before localization.
```

Export localizes per hash collection/chunk:

```text
global bone -> chunk local bone
```

The exported `Blend.buf` uses compact chunk local bone indices starting at zero. `PaletteTable.buf` maps those chunk local indices back to global capture-store bone indices.

Hard limit:

```text
chunk local bone count <= 256
```

This limit applies per hash collection/chunk after recursive object gathering and bone localization.

### Import Into Root

Import Selected IBs places imported objects directly under the single root collection. It does not create a separate import root.

After Build Global Bone Pool, users manually move/edit/copy objects into the desired hash subcollections for export.

The addon may provide a convenience button:

```text
Sort Selected By Object Hash
```

This button reads each selected object's first independent 8-hex source hash, finds or creates the matching hash collection under the root, and moves/links the object there. This is only a workflow accelerator. The final export identity is still determined by the destination hash collection.

### Candidate List And Rename

Global bone mapping is driven by a UI list, not by an extra Blender collection. Too many collections make the workflow harder to understand.

The Candidate IB list is the single editable source for import and global-pool building. Analyze fills it automatically from FrameAnalysis. Users can add/remove entries manually, and a refresh action can merge source hashes from objects currently under the root collection.

For collection refresh and compatibility renaming, the object source hash is the first independent 8-hex value in the object name or stored custom property. This is used only for refreshing/building/renaming the source mapping list, not for export identity.

The scanner uses the source hash to find current FrameAnalysis data:

```text
source IB hash
match_index_count
source_index_count
source local bone count
runtime validation fields
```

Build Global Bone Pool creates:

```text
source local bone -> global bone
capture records sorted by match_index_count descending
CaptureMeta / global pool manifest entries
the single export root collection, if missing
```

The pool build may immediately rename matched source objects into global vertex-group semantics. A separate Apply Global Bone Names button must also remain available for objects not imported by this plugin.

Rename logic:

```text
object source hash + object local bone index -> global bone index
```

Scan must not silently rewrite meshes.

This rename function is an important compatibility feature. For external models imported by other tools, the plugin can still convert local numeric groups to the global pool naming scheme by using the object hash and the built source mapping table.

### Import/Export/INI Rules

Import, export, and base INI generation should follow `E:/vscode/mod_importer`.

Notable inherited rules:

```text
Analyze picks the resolved draw/model using mod_importer discovery logic.
Import should prefer the output/display path with the strongest o-slot evidence where that is the mod_importer rule.
Texture marking should reuse mod_importer candidate and material semantics.
Export should use mod_importer collection properties and buffer writers.
INI generation should reuse mod_importer fast-path syntax where possible.
```

BoneMerge adds only the global bone capture/replay tables and the delayed shadow replay scheduling on top of the shared import/export base.

Texture overrides are the exception to direct INI reuse: candidate detection and texture conversion can reuse `mod_importer`, but replacement output uses `TextureOverride hash = <texture_hash>; this = Resource...` rather than slot-style `ps-t` assignment.

## Analyze And Import Flow

The UI should expose three core build buttons before export:

```text
Analyze
Import Selected IBs
Build Global Bone Pool
```

LOD projects add one optional button after the global pool exists:

```text
Analyze LOD
```

### Analyze

Analyze follows `mod_importer` where possible, with BoneMerge-specific discovery added.

Purpose:

```text
find visible/g-buffer anchors for the target character
walk back to the early shadow/capture region from those anchors
rediscover the current frame's two shadow VS hashes without hardcoding them
find unique importable IB/VB geometry across the frame
mark which candidates have matching early-shadow capture data
collect draw metadata needed for import, texture marking, and pool building
populate an editable candidate IB list
```

Suggested discovery:

```text
1. Parse FrameAnalysis log and dumped draw files.
2. Find visible/g-buffer-like draws for the target character by output/RT evidence and IB/VB mesh identity.
3. Use those visible IB slices to locate the earlier matching shadow/capture draws.
4. Identify the two shadow VS hashes from that early shadow/capture region.
5. Traverse all FrameAnalysis draws and merge unique importable IB/VB geometry into candidate entries.
6. For each candidate, choose one import source draw, preferring the strongest visible/material draw and falling back to any valid dump with complete VB0-VB3/IB data.
7. For each candidate, separately record whether it has matching early-shadow hits under the discovered shadow VS pair.
8. Fill the Candidate IB list.
```

G-buffer/display draws are the most stable anchor for resolving the target character and finding the current shadow VS pair, because they expose visible material geometry without requiring shader hashes to stay constant across updates. They are not the only import source: transparent objects may not have a g-buffer pass, and some valid meshes may only be available through other material/effect passes. The final import set is therefore the editable Candidate IB list, not "every draw that used a particular VS".

The two shadow VS hashes discovered during Analyze are frame-local filters for capture and delayed shadow replay. They must not become persistent identity. If a game update changes them, a fresh Analyze should rediscover them from the current FrameAnalysis by matching g-buffer/display IBs back to the early shadow/capture stage.

For vertex/index payloads, raw `.buf` dumps are the source of truth. Expanded text dumps and shader disassembly are layout evidence, but actual import data must be sliced from `.buf` with the draw's recorded offset, stride, index count, vertex count, and IA format. This is important because a shared backing buffer can contain many parts, and a deduped or expanded view can appear correct while starting from the wrong byte range.

Users may add/remove candidate IBs before import. This supports partial imports where the user only wants one or a few IBs. The UI may also refresh the Candidate IB list from the current collection, reading object metadata or hash-like names and merging those IBs into the same list.

### Analysis Manifest

Analyze produces one manifest. Import, Build Global Bone Pool, Generate, and Inject should consume this manifest instead of rescanning FrameAnalysis independently.

Top-level fields:

```json
{
  "schema_version": 1,
  "frameanalysis_dir": "",
  "target": {},
  "shadow_stage": {},
  "candidate_ibs": [],
  "draw_hits": [],
  "texture_candidates": [],
  "producer_dispatches": [],
  "bone_pool_order": [],
  "lod_frameanalysis": [],
  "lod_links": [],
  "lod_capture_records": [],
  "validation": []
}
```

#### target

The target section describes the automatically resolved frame anchors for Analyze. It is diagnostic output, not a required manual IB input.

```json
{
  "visible_anchor_ibs": [
    "ae1ab184"
  ],
  "selection_mode": "auto_gbuffer_anchor"
}
```

`selection_mode` is diagnostic only. It should describe whether the anchor came from g-buffer/display detection, collection refresh, or a manual candidate entry.

#### shadow_stage

The shadow stage is the source of native bone palettes and delayed shadow replay scheduling. It is not the only source of import geometry.

```json
{
  "shadow_vs_hashes": [
    "transparent_or_late_shadow_vs",
    "normal_shadow_vs"
  ],
  "stage_draw_start": 0,
  "stage_draw_end": 0,
  "transparent_vs_hash": "",
  "normal_vs_hash": "",
  "host_draw_index": 0,
  "host_ib_hash": "",
  "host_match_index_count": 0
}
```

`shadow_vs_hashes` is the required capture filter for this FrameAnalysis. `transparent_vs_hash` and `normal_vs_hash` are optional classifications used later by replay scheduling. If classification is ambiguous, keep both VS hashes in `shadow_vs_hashes` and leave the classification fields empty.

#### draw_hits

`draw_hits` records the evidence used by candidates. A candidate can have both import hits from visible/material passes and capture hits from the early shadow stage. Only early-shadow hits whose VS hash matches `shadow_stage.shadow_vs_hashes` are valid bone-palette producers.

```json
{
  "draw_index": 314,
  "ib_hash": "ae1ab184",
  "first_index": 190539,
  "index_count": 8148,
  "base_vertex": 0,
  "vs_hash": "",
  "ps_hash": "",
  "vb0_hash": "",
  "ib_dump_path": "",
  "vb_dump_paths": {},
  "vs_t0_hash": "",
  "vs_t0_path": "",
  "cb1_path": "",
  "cb1_palette_current": 1024,
  "cb1_palette_previous": 1024,
  "producer_dispatch_index": 274,
  "pass_role": "shadow_unknown",
  "use_role": "capture"
}
```

`pass_role` is one of `transparent_shadow`, `normal_shadow`, `visible_material`, `effect`, or `unknown`. It is a scheduling and material hint, not import identity. `use_role` is one of `import`, `capture`, or `both`.

#### candidate_ibs

Candidate IBs are the editable UI list. They are merged from `draw_hits` by draw slice identity.

```json
{
  "enabled": true,
  "ib_hash": "ae1ab184",
  "match_first_index": 190539,
  "match_index_count": 8148,
  "display_name": "ae1ab184-8148-190539",
  "draw_indices": [314, 318, 349, 357],
  "import_draw_index": 357,
  "import_vs_hash": "",
  "import_ps_hash": "",
  "shadow_capture_ready": true,
  "shadow_draw_indices": [314, 318],
  "source_index_count": 8148,
  "producer_dispatch_index": 274,
  "local_bone_count": 16,
  "texture_region_key": "ae1ab184-8148-190539",
  "import_paths": {
    "ib": "",
    "vb": {},
    "layout": ""
  }
}
```

`enabled` is user-editable. Disabled candidates stay in the manifest for diagnostics but are not imported and do not build the global pool.

`shadow_capture_ready` is the gate for Build Global Bone Pool. A candidate can be importable even when this field is false, but it must be excluded from the global bone pool because the runtime cannot safely capture its native palette from the early shadow stage.

#### texture_candidates

Texture candidates reuse `mod_importer` semantics and may come from g-buffer/display draws, shadow draws, or visible material draws.

```json
{
  "region_key": "ae1ab184-8148-190539",
  "draw_index": 357,
  "ps_hash": "",
  "rt_count": 3,
  "slot": "ps-t7",
  "hash": "",
  "source_path": "",
  "semantic_hint": "base_color"
}
```

Texture candidates do not decide model import membership. They provide material choices for marked textures and later texture-hash replacement overrides. The `slot` field is retained as evidence for semantic classification, not as the generated INI binding target.

#### producer_dispatches

Producer dispatches describe the native bone-palette writes that can be captured into the global store.

```json
{
  "producer_dispatch_index": 274,
  "collect_key_value": 46932,
  "start_vertex": 46932,
  "vertex_count": 1827,
  "cs_hash": "",
  "u0_hash": "",
  "u1_hash": "",
  "vs_t0_hash": "",
  "t0_path": "",
  "local_bone_count": 16
}
```

The dispatch order used for global-pool base assignment is:

```text
match_index_count descending for source candidate grouping
then collect_key_value/start_vertex ascending inside one source buffer chain
then producer_dispatch_index as tie-breaker
```

This preserves the runtime bone-store semantics and avoids event-order drift when a later event writes an earlier vertex segment.

#### bone_pool_order

`bone_pool_order` is derived from enabled candidates and producer dispatches. It is the input for Build Global Bone Pool.

```json
{
  "ib_hash": "ae1ab184",
  "match_first_index": 190539,
  "match_index_count": 8148,
  "producer_dispatch_index": 274,
  "global_bone_base": 378,
  "local_bone_count": 16,
  "capture_store_base": 378
}
```

The global bone index and capture-store index have the same meaning in HLSL. Exported chunk palettes must reference these indices exactly.

#### lod_frameanalysis

`lod_frameanalysis` records optional LOD scan sources. They are analyzed only after the main global pool exists.

```json
{
  "lod_level": 1,
  "path": "E:/XXMI/EFMI/FrameAnalysis-2026-05-05-225007",
  "log_path": "E:/XXMI/EFMI/FrameAnalysis-2026-05-05-225007/log.txt",
  "shadow_vs_hashes": ["", ""],
  "stage_draw_start": 0,
  "stage_draw_end": 0
}
```

#### lod_links

`lod_links` maps canonical main source parts to resolved LOD source parts. The link is primarily discovered by VS/PS/pass evidence, then validated by texture slots and structure. Hashes are the result of the link, not the primary search key.

```json
{
  "lod_level": 1,
  "main_ib_hash": "640d1c0e",
  "main_match_index_count": 46845,
  "main_first_index": 0,
  "lod_ib_hash": "lodhash00",
  "lod_match_index_count": 12345,
  "lod_first_index": 0,
  "pass_role": "normal_shadow",
  "main_vs_hash": "",
  "lod_vs_hash": "",
  "main_ps_hash": "",
  "lod_ps_hash": "",
  "texture_slot_match": "exact",
  "confidence": "structural_exact"
}
```

#### lod_capture_records

`lod_capture_records` are runtime scatter capture records. They tell `RecordBonesScatter` how a LOD local palette fills the canonical high-detail global pool.

```json
{
  "lod_level": 1,
  "lod_ib_hash": "lodhash00",
  "lod_match_index_count": 12345,
  "lod_first_index": 0,
  "lod_local_bone_count": 64,
  "pair_base": 0,
  "pair_count": 96,
  "capture_vs_filter": 202,
  "capture_draw_indices": [500],
  "pairs": [
    {"lod_local_bone": 12, "canonical_global_bone": 45},
    {"lod_local_bone": 12, "canonical_global_bone": 46}
  ]
}
```

The inline `pairs` array is for diagnostics and project portability. The generated runtime should materialize it into compact static buffers such as `LodCaptureMeta.buf` and `LodCapturePairs.buf`.

#### validation

Analyze should record warnings and hard failures instead of hiding them in console output.

```json
{
  "severity": "warning",
  "code": "ambiguous_shadow_vs_role",
  "message": "Two shadow VS hashes were found, but transparent/normal classification is not stable.",
  "draw_indices": [300, 301]
}
```

Hard failures block Import/Build/Generate. Warnings allow the user to continue but should remain visible in compact status and in the diagnostics log.

### Import Selected IBs

Import consumes the Candidate IB list. For each selected IB it imports the corresponding model and attributes using the mod_importer-compatible logic.

The imported data format is driven by the union of all IA layouts observed for the candidate IB's draw hits. A single IB can be rendered by several passes that bind the same IB/VB slices but interpret the same bytes with slightly different semantic names. Import must therefore preserve the buffer layout that satisfies every selected pass, not only the most visible material pass.

For each imported candidate, the manifest should store the resolved IA contract:

```json
{
  "ib": {
    "hash": "640d1c0e",
    "format": "DXGI_FORMAT_R16_UINT",
    "offset": 12722752,
    "index_count": 46845,
    "first_index": 0,
    "base_vertex": 0,
    "max_index": 13570,
    "unique_vertex_count": 13571
  },
  "vertex_count": 13571,
  "buffers": {
    "position": {
      "vb_slot": 0,
      "hash": "55cdcf9e",
      "offset": 12071296,
      "stride": 16,
      "fields": [
        {"semantic": "POSITION0", "offset": 0, "format": "R32G32B32_FLOAT"},
        {"semantic": "NORMAL0", "offset": 12, "format": "R32_FLOAT", "packed": true}
      ]
    },
    "texcoord": {
      "vb_slot": 1,
      "hash": "69165990",
      "offset": 12288448,
      "stride": 20,
      "fields": [
        {"semantic": "TEXCOORD0", "offset": 0, "format": "R32G32_FLOAT"},
        {"semantic": "TEXCOORD1", "offset": 8, "format": "R32G32_FLOAT"},
        {"semantic": "TEXCOORD4", "offset": 16, "format": "R8G8B8A8_SNORM"}
      ]
    },
    "blend": {
      "vb_slot": 2,
      "hash": "8e3617eb",
      "offset": 12559888,
      "stride": 12,
      "fields": [
        {"semantic": "BLENDWEIGHTS0", "offset": 0, "format": "R16G16B16A16_UNORM"},
        {"semantic": "BLENDINDICES0", "offset": 8, "format": "R8G8B8A8_UINT"}
      ]
    },
    "position_alias": {
      "vb_slot": 3,
      "aliases": "position",
      "fields": [
        {"semantic": "TEXCOORD5", "offset": 0, "format": "R32G32B32_FLOAT"},
        {"semantic": "TEXCOORD6", "offset": 12, "format": "R32_FLOAT", "packed": true}
      ]
    }
  }
}
```

The `640d1c0e` LOD sample confirmed these concrete rules:

```text
positions
packed normals / auxiliary packed frame data
blend indices and weights
UV0
UV1
packed TEXCOORD4 auxiliary data
texture/material marks
source hash metadata
draw slice metadata
```

#### Imported Buffer Semantics

`Position` buffer:

```text
stride = 16
+0  POSITION0  R32G32B32_FLOAT
+12 NORMAL0    R32_FLOAT
```

`NORMAL0` is a packed octahedral normal payload carried through an IA `R32_FLOAT` declaration. Import should decode it with a BoneMerge-specific implementation of the game's octahedral normal routine and assign the result to Blender custom split normals, so the imported model's visible normals match the game. Export should re-encode from the Blender custom normals back into the same packed format. This codec must not assume `mod_importer`'s normal packing or shader path, because that project does not use this octahedral normal format. For untouched imported meshes, preserving the original raw payload as a fallback is allowed, but the design contract is that Blender's custom normals are the editable truth.

`vb3` can be the same physical slice as `vb0`. In the tested main material pass, `vb3` binds the same `55cdcf9e` buffer and the same offset/stride as `vb0`, but the shader reads it as:

```text
+0  TEXCOORD5  R32G32B32_FLOAT
+12 TEXCOORD6  R32_FLOAT
```

Do not create a separate imported data stream for `vb3` when it aliases `vb0`. Preserve the alias in the manifest and generate `vb3 = ref Position` on export.

`Texcoord` buffer:

```text
stride = 20
+0   TEXCOORD0  R32G32_FLOAT       UV0
+8   TEXCOORD1  R32G32_FLOAT       UV1
+16  TEXCOORD4  R8G8B8A8_SNORM     packed auxiliary data
```

`UV1` is real source data, not filler. For `640d1c0e`, the first vertices contain values such as:

```text
0.591714621, 0.185148418
0.594583273, 0.177773193
0.587006032, 0.184763551
0.0254230108, 0.334441602
0.0340663977, 0.35172838
```

Import should create a second Blender UV layer for `TEXCOORD1`, for example `UVMap_1`. If the user later exports a mesh with only one UV layer, export may copy `UV0` into `UV1` as a compatibility fallback, but imported native models should keep the original `UV1` values.

`TEXCOORD4` is not a UV layer. It is a packed signed-normalized auxiliary field. It is used by multiple passes under different semantic names, for example `TEXCOORD4` and sometimes `TEXCOORD2` at the same byte offset. Import should keep it as a custom packed attribute or raw byte payload, not as a Blender UV map.

`Blend` buffer:

```text
stride = 12
+0  BLENDWEIGHTS0  R16G16B16A16_UNORM
+8  BLENDINDICES0  R8G8B8A8_UINT
```

`BLENDINDICES0` are source-local palette indices. They are not global bone indices. Import should keep them as local vertex groups until Build Global Bone Pool renames or maps them. Export localizes the final chunk again and writes compact local indices, while `PaletteTable.buf` maps those final local indices back to global capture-store indices.

The source `IB` can be `DXGI_FORMAT_R16_UINT`, even if older exported mods wrote `DXGI_FORMAT_R32_UINT`. Import should record the native index format from FrameAnalysis. Export may choose a wider format only when the generated index data requires it or when reusing an inherited mod_importer rule.

#### Multi-Pass Layout Union

The importer must gather layouts from every enabled draw hit for one candidate IB. For the tested `640d1c0e` group, the same core slices were reused across all hits:

```text
ib  = 640d1c0e
vb0 = 55cdcf9e
vb1 = 69165990
vb2 = 8e3617eb
vb3 = 55cdcf9e only on the full material path
```

Different passes can expose different semantic views over those same bytes. Examples:

```text
main material path:
  vb1 +0  TEXCOORD0
  vb1 +16 TEXCOORD4
  vb3 aliases vb0 as TEXCOORD5/TEXCOORD6

later outline/shadow-like paths:
  vb1 +0  TEXCOORD0
  vb1 +8  TEXCOORD1
  vb1 +16 TEXCOORD4 or TEXCOORD2
```

Therefore the imported mesh data model for this kind of part must keep `vb1` as a 20-byte stream with both `UV0` and `UV1`, even when the visible material shader only reads `UV0`.

#### Blender WYSIWYG Export Contract

Blender is the final editable truth. Export should match what the user sees in Blender rather than hidden original-buffer state.

Rules:

```text
Use the evaluated mesh from Blender/mod_importer-compatible export rules.
Apply visible modifiers and current shape-key results the same way mod_importer does.
Export positions from the evaluated mesh.
Export UV0 from the primary UV layer.
Export UV1 from the second UV layer; if absent, copy UV0.
Export normals from Blender custom split normals, then octahedral-encode them for NORMAL0.
Keep packed non-UV auxiliary fields such as TEXCOORD4 as custom/raw attributes.
If a required raw auxiliary attribute is missing on an edited mesh, use the mod_importer-compatible fallback or stop with a diagnostic instead of writing garbage.
```

Shape key and morph handling should be reused from `E:/vscode/mod_importer` wherever possible. BoneMerge should not invent a separate shape-key interpretation. The user-facing rule remains: the mesh visible in Blender is the mesh that is written.

### Build Global Bone Pool

Build Global Bone Pool consumes the editable Candidate IB list. There is no separate mapping collection or preset list in the simplified workflow.

The accepted pool inputs are:

```text
enabled candidate
shadow_capture_ready == true
local_bone_count > 0
```

Candidates that were imported only for editing/reference but cannot be matched to the early shadow/capture stage are skipped. This keeps the runtime tables honest: every global-pool source must correspond to native matrices that can actually be captured before delayed replay.

Accepted candidates are sorted by descending `match_index_count`, with larger source chunks occupying earlier global-bone positions and earlier vertex-group naming ranges. Ties use local bone count, mesh fingerprint, first index, and hash only as deterministic tie-breakers.

It also creates the single export root collection if it does not exist. After this step, users edit models and place final meshes under hash collections inside the export root.

## LOD Mapping And Runtime

LOD support uses the main/global-pool model as the canonical skeleton. LOD does not define a second final mesh contract. Instead, LOD native draws provide distance-specific native bone matrices that are scattered into the same canonical global bone pool used by the high-detail replacement model.

The user may select:

```text
Main FrameAnalysis
LOD FrameAnalysis 0..N
```

The main FrameAnalysis is the source of truth for:

```text
candidate IB list
imported source models
canonical global pool built from enabled capture-ready candidates
canonical global bone indices
final export chunks and PaletteTable semantics
```

LOD FrameAnalysis folders are optional secondary sources. They are analyzed only after Build Global Bone Pool, because LOD mapping needs the canonical global bone indices created from the user-confirmed Candidate IB list.

### LOD Flow

The intended user flow is:

```text
1. Analyze Main.
2. Import selected main candidates.
3. User deletes unwanted candidates from the Candidate IB list or refreshes the list from collection objects.
4. Build Global Bone Pool from enabled, shadow-capture-ready candidates.
5. Analyze LOD.
6. Generate runtime buffers and INI.
```

This keeps one editable list as the UI source of truth:

```text
Candidate IB List:
  all main candidates discovered from FrameAnalysis, plus manual or collection-refreshed entries
  enabled entries control import
  enabled + shadow_capture_ready entries control Build Global Bone Pool
```

The LOD analyzer must consume the built global pool, not the full Candidate IB list. Imported candidates that the user deleted, disabled, or that failed `shadow_capture_ready` must not pollute LOD matching.

### Main To LOD Draw Linking

Main and LOD IB hashes usually differ. Hash is therefore not the primary relationship key between a main part and a LOD part.

If a LOD and the main part have the same IB hash, the analyzer must still validate them structurally. The safe cases are:

```text
same hash + same match/index range + same IA/VB fingerprint + same local palette shape:
  treat as the same runtime identity; no separate LOD scatter table is needed

same hash but different geometry, VB fingerprint, local bone count, or palette shape:
  treat as a distinct LOD candidate and require a reliable runtime discriminator
```

Reliable discriminators include `match_index_count`, first index/range where supported by the generated override pattern, VS/PS/pass role, and texture-slot evidence. If the same hash and same override filters can match two incompatible identities, generation must stop with an ambiguity diagnostic. A single runtime override must not sometimes run the main linear capture and sometimes need LOD scatter capture without a way to tell which draw it is seeing.

For each main source IB in the Global Pool list, build a main draw signature:

```text
main_ib_hash
main_match_index_count
main_first_index
main_vs_hashes by pass role
main_ps_hashes by pass role
main_ps_texture_slot_hashes
main_ia_layout_union
main_vertex_count
main_index_count
main_position_bounds
main_blend_index_count
main_mesh_fingerprint
```

Then search each LOD FrameAnalysis for candidate LOD draws:

```text
1. Prefer draws with the same or corresponding VS hash and same pass role.
2. If VS is shared, compare PS hash.
3. If VS and PS are both shared, compare PS texture slot hashes.
4. Use IA layout, index_count, vertex_count, bounds, and sampled mesh fingerprints as structural validation.
5. Use IB hash only as an output identity after the relationship is resolved.
```

The resolved relationship should be recorded explicitly:

```json
{
  "main_ib_hash": "640d1c0e",
  "main_match_index_count": 46845,
  "main_first_index": 0,
  "lod_ib_hash": "lodhash00",
  "lod_match_index_count": 12345,
  "lod_first_index": 0,
  "pass_role": "normal_shadow",
  "vs_match": "same_hash",
  "ps_match": "same_hash",
  "texture_slot_match": "exact",
  "confidence": "structural_exact"
}
```

If multiple LOD candidates remain after VS/PS/texture/layout checks, do not silently pick one. Keep the candidates in diagnostics and require manual confirmation or a stronger structural match.

### Whole-Set Spatial Mapping

LOD mapping must not assume one main IB maps to exactly one LOD IB. Character part splitting can change between LOD levels. For example:

```text
main:
  part A = thigh + waist

LOD:
  part A = thigh
  part B = waist
```

Therefore the mapper should combine all user-confirmed main Global Pool objects into one `MainSet`, and combine all resolved LOD candidate parts for the same character into one `LodSet`.

```text
MainSet:
  vertices from all Global Pool source objects
  source local bone weights
  source local bone -> canonical global bone
  position / packed normal / optional UVs

LodSet:
  vertices from resolved LOD IB candidates
  LOD local bone weights
  LOD IB hash and match_index_count
  position / packed normal / optional UVs
```

The mapping algorithm should use weighted nearest-neighbor voting across the whole sets:

```text
for each LOD vertex:
    find K nearest MainSet vertices in object space
    score each neighbor by:
        distance similarity
        optional normal similarity
        optional UV similarity
        part/bounds compatibility

    for each LOD local bone affecting the LOD vertex:
        for each main global bone affecting each matched main vertex:
            vote[lod_ib, lod_local_bone, main_global_bone] +=
                lod_weight * main_weight * match_score
```

The result is not a one-to-one mapping. The analyzer should produce:

```text
lod_local_to_global:
  best canonical global bone for a LOD local bone, diagnostic and rename support

global_to_lod_local:
  best LOD IB + LOD local bone that should fill a canonical global bone at runtime
```

`global_to_lod_local` is the runtime-critical table and may be one-to-many in the reverse direction:

```text
LOD part A local bone 12 -> global bone 45
LOD part A local bone 12 -> global bone 46
LOD part A local bone 12 -> global bone 47
LOD part B local bone 3  -> global bone 80
```

This lets a lower-detail LOD skeleton fill a complete high-detail canonical global pool. Missing high-detail bones are approximated by the nearest stable LOD local bone chosen by the vote table.

### LOD Runtime Tables

LOD capture uses scatter writes into the canonical global bone pool. It must write both current and previous matrices.

Suggested runtime resources:

```text
Buffer/LodCaptureMeta.buf
uint4 LodCaptureMeta[N]

x = pair_base
y = pair_count
z = lod_local_bone_count
w = flags/reserved

Buffer/LodCapturePairs.buf
uint2 LodCapturePairs[K]

x = lod_local_bone
y = canonical_global_bone
```

Main capture remains linear or manifest-ordered:

```text
main local bone -> canonical global bone
```

LOD capture uses scatter:

```text
for pair in LodCapturePairs[pair_base : pair_base + pair_count]:
    global_current[pair.canonical_global_bone] =
        native_lod_t0_current[pair.lod_local_bone]

    global_previous[pair.canonical_global_bone] =
        native_lod_t0_previous[pair.lod_local_bone]
```

The draw/consume path stays unchanged:

```text
chunk local bone -> PaletteTable -> canonical global bone -> local fake vs-t0
```

This is the key LOD contract: final replacement geometry still uses the high-detail canonical global bone semantics. LOD only changes which native source matrices fill those global slots before drawing.

### LOD TextureOverrides And INI Output

Generated INI must include `TextureOverride` sections for resolved LOD IB hashes and `match_index_count` values. The logic is the same shape as the main path, but the capture shader and metadata are LOD scatter-aware.

For each resolved LOD source IB:

```text
TextureOverride LOD capture:
  hash = lod_ib_hash
  match_index_count = lod_match_index_count
  if vs == expected_lod_capture_vs:
      bind LodCaptureMeta one-record view
      bind LodCapturePairs
      bind native vs-t0
      dispatch RecordBonesScatter
```

For replacement drawing at LOD distance:

```text
TextureOverride LOD consume/replay:
  hash = lod_ib_hash
  match_index_count = lod_match_index_count
  if this LOD source has exported replacement geometry:
      skip original LOD draw where appropriate
      gather canonical global bones into the local fake vs-t0
      draw the high-detail replacement chunks that target this LOD stage
```

Unmodified LOD source IBs are left on the game's original path. Modified/exported LOD source IBs follow the same delayed shadow replay rule as main parts:

```text
1. Capture all required LOD native palettes into the canonical global pool.
2. Skip only original LOD shadow draws whose source IBs are replaced.
3. Leave unmodified LOD shadow draws untouched.
4. At the final compatible LOD shadow host, replay all delayed modified replacement chunks.
5. Draw transparent shadow batches first, then bind white shadow PS resources and draw normal shadow batches.
```

The replay host and resource bindings are LOD-stage-specific, but the replacement chunk resources, chunk local palettes, and `PaletteTable` still use canonical high-detail global bone semantics.

## Replay Scheduling

The core replay goal is that any exported chunk can use bones captured from any source IB. A modified IB may reference bones from draws that occur later than its original shadow draw. Drawing that modified IB at its original shadow position can therefore read incomplete bone data.

To avoid this, modified shadow drawing is delayed:

```text
1. Early/source encounters capture native bone palettes.
2. Original shadow draws for modified/exported IBs are skipped at their original positions.
3. Unmodified game IBs are left alone and keep their original shadow drawing.
4. At the final compatible shadow host, all delayed modified shadow chunks are replayed after the global bone store is complete.
```

The replay host is not limited to drawing only its own IB. It is a late scheduling point that can draw replacement chunks from multiple source IBs, because those chunks now gather from the completed global bone store.

Replay host selection:

```text
host_draw_index > max(required_capture_draw_index)
host is the last matching draw in the compatible render-state group
host pass has render/depth/blend/texture state that replacement chunks can inherit
```

The last matching draw is preferred because the global bone store is complete by then and the current render state is compatible with the final replacement drawing path.

Skip rules:

```text
if source IB has exported replacement shadow geometry:
    skip that source IB's original shadow draw at its original shadow pass
else:
    leave the original game shadow draw untouched
```

This keeps untouched game parts, such as original hair, on the game's normal shadow path while delaying only the parts that need merged bones.

The "last" host is calculated from the target character's early shadow stage, not from the whole frame. Starting from the selected target IB, Analyze finds the corresponding shadow-stage draw cluster and its two shadow VS hashes. It then collects shadow-stage hits for the candidate/source IB set and chooses the maximum draw index in that cluster as the replay scheduling point.

In the observed pattern, the earlier VS in the pair is the normal/opaque shadow path and the later VS is the transparent shadow path. When the last host is the later transparent VS draw, replay uses that state as the scheduling point:

```text
1. replay transparent shadow chunks first, without binding white PS resources
2. bind the white shadow PS resource once
3. replay normal/opaque shadow chunks
```

Normal/opaque chunks need the white PS resource because they are being delayed from the earlier normal shadow draw into the later transparent host. Transparent chunks inherit the transparent shadow state and do not bind white.

The host can be any compatible late draw in the cluster. It does not have to be the same IB as every replayed chunk. If the host IB itself is exported and skipped, the override still acts as the scheduling carrier; if the host IB is unmodified, its original draw should remain and the replay draws are appended by the generated override logic.

When a replay host draws multiple chunks with one reusable local bone buffer, each chunk must gather and draw immediately:

```text
Gather chunk A -> bind local vs-t0 -> draw chunk A
Gather chunk B -> bind local vs-t0 -> draw chunk B
```

Do not gather several chunks first and draw later, because the local buffer is overwritten by each gather call.

### Shadow And Transparent Replay

Some characters have an earlier group with two relevant VS hashes. The later draw hash in that group can represent the transparent shadow path. At the replay host, transparent shadow geometry should be drawn first. Then the PS slots should be changed to the white shadow resources and the normal shadow geometry should be drawn.

The replay plan must support ordered draw batches:

```text
1. transparent shadow batch
2. set white shadow PS resources
3. normal shadow batch
4. other visible batches, if applicable
```

This ordering belongs in the generated replay manifest, not in an ad hoc UI choice.

## Implementation Readiness

The core architecture is ready for implementation. Remaining items are verification gates, not blockers for starting the plugin.

Implementation should not be constrained by the old plugin architecture. The old Scan/BoneStore/preset code can be used as reference material, but the v2 path should be allowed to replace modules outright when the old shape conflicts with the new data contract. Compatibility is useful only when it does not compromise the main flow:

```text
Analyze Main -> Candidate IB List -> Import Selected -> Build Global Bone Pool -> Export -> Generate INI
```

Legacy panels/operators may remain temporarily while the v2 path is built, but they are not architectural requirements.

### Verification Gates

Before claiming the first version is correct, validate these points against real FrameAnalysis samples:

```text
Octahedral normal codec:
  derive the exact pack/unpack bit layout from the game shader and round-trip imported normals.

FrameAnalysis slicing:
  read vertex/index data from .buf with draw-local offset/stride/count and verify index bounds.

Texture hash replacement:
  collect source texture hashes from sampled resources and emit TextureOverride hash = texture_hash, this = Resource...

White shadow resource:
  identify the exact resource(s) and slot behavior used for delayed normal/opaque shadow replay.

Replay host:
  verify that the chosen last shadow-stage host occurs after every required capture draw and inherits compatible state.

Vertex group export:
  prune empty groups, remap groups from zero, enforce <= 256 local groups per exported collection/chunk, and write PaletteTable from local group to canonical global bone.

Raw auxiliary attributes:
  preserve required packed fields such as TEXCOORD4, or stop export with a clear diagnostic when an edited mesh lacks them.

LOD:
  keep optional until the main path is stable; same-hash LOD/main cases require structural validation and a runtime discriminator.
```

### First Implementation Slice

Build the plugin as a vertical slice instead of trying to finish every feature at once:

```text
1. Manifest schema and diagnostics.
2. Main FrameAnalysis parser:
   shadow VS pair discovery, draw hits, candidate IB list, .buf slicing metadata.
3. Import Selected IBs:
   positions, UV0, UV1, blend weights/indices, octahedral custom normals, raw auxiliary attributes.
4. Global Pool UI list:
   add selected objects, build source-local to canonical-global mapping, rename groups.
5. Export root contract:
   one root collection, hash subcollections, WYSIWYG evaluated mesh export, <= 256 local bones.
6. Static runtime buffers:
   CaptureMeta, PaletteMeta, PaletteTable, global/local current and previous matrix stores.
7. INI generation:
   capture overrides, gather/replay overrides, texture hash overrides, delayed shadow replay plan.
8. LOD analysis and scatter capture:
   add after the main non-LOD loop can import, export, and replay correctly.
```

The first usable milestone should be:

```text
Analyze Main -> Import Selected -> Build Global Pool -> Export one edited hash collection -> generated INI renders without deformation.
```
