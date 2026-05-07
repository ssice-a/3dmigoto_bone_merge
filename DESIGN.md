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
- FrameAnalysis CB dumps are raw full-buffer dumps. The shader-visible `cb1[5]` for a draw is located at `raw_cb1[first_constant + 5]`.
- The logical `cb1` content can differ per pass, and the raw CB rows used by each pass can differ.
- The bone-window pointer fields used by current capture logic, shader-visible `cb1[5].x` and `cb1[5].y`, remained stable for the same tested part IB across multiple passes after applying `first_constant + 5`.
- `cb1[5].z` differed between normal and afterimage/effect passes, for example `258`, `2`, and `1`.

For example, in `FrameAnalysis-2026-05-05-222451`, the tested `640d1c0e` passes used different `first_constant` values but all resolved to shader-visible `cb1[5].xy = 449614/448066`; tested `2009f0d6` passes resolved to `450751/448828`.

Therefore the design must not assume that the whole `cb1` is identical for all passes of one IB. Capture must use the current draw's shader-visible `cb1[5].x/y` to read the native bone window. Consume must preserve the current draw's `cb1[5].z/w` and rewrite only `cb1[5].x/y` when redirecting to the local rebuilt palette.

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

`cb1[5].x/y` are both offline validation fields and runtime capture inputs. They are native bone-window offsets into the currently bound game `vs-t0` store. They must not be assumed to be `0/1024` during capture.

When the consume path binds `ResourceLocalFakeT0_SRV` as `vs-t0`, the draw no longer points at the game's native large bone store. The draw's shader-visible `cb1[5].x/y` must therefore be redirected to the local fake layout:

```text
cb1[5].x = 0
cb1[5].y = LOCAL_PREVIOUS_ROW_OFFSET
```

For the default local fake store, `LOCAL_PREVIOUS_ROW_OFFSET = 1024`.

The final consume identity is:

```text
final_region_ib_hash
match_index_count
match_first_index
part_index
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

Do not generate one INI `ResourceBoneMeta_*` section per IB. Per-IB INI resources make the runtime config noisy and were a failed older design. Main capture metadata and source-local to global mapping should be materialized into one compact static buffer file:

```text
Buffer/MainCaptureBoneMap.buf
```

The static main capture map format is:

```text
uint MainCaptureBoneMap[]

header:
  x = record_count
  y = pair_table_uint_base
  z = total_pair_count
  w = flags/reserved

record:
  x = pair_base
  y = pair_count
  z = source_bone_count
  w = flags/reserved

pair:
  x = source_local_bone
  y = target_global_bone
```

At runtime, the `TextureOverride` hash and `if vs == ...` filter select the capture draw, and `x100` selects the compact capture record inside the static table.

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
  "native_cb1_xy": [449614, 448066],
  "redirect_cb1_xy": [0, 1024],
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
  "main_capture_bone_map_path": "Buffer/MainCaptureBoneMap.buf",
  "lod_capture_bone_map_path": "Buffer/LodCaptureBoneMap.buf",
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

The replay host is a late compatible shadow draw used to replay all delayed modified shadow parts after the global bone store is complete.

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
      "parts": []
    },
    {
      "name": "normal_shadow",
      "set_white_ps_t0": true,
      "parts": []
    }
  ]
}
```

Each part entry should include the exported mesh resources, draw calls, and palette record:

```json
{
  "source_ib_hash": "2e5d9294",
  "match_index_count": 23220,
  "part_index": 0,
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
  "native_cb1_xy_consistent_by_part": true,
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
  run = CustomShader_ExtractCB1
  cs-t2 = ResourceMainCaptureBoneMap
  run = CustomShader_RecordBones
endif
```

`RecordBones` reads the current draw's native `vs-t0` by reference and reads the current draw's shader-visible `cb1[5].x/y` from the extracted CB1 copy.

FrameAnalysis showed that the game binds a large native bone store in `vs-t0`. The native `cb1[5].x/y` values point into that large store and are not `0/1024`. Therefore capture must use:

```text
native_current_base  = cb1[5].x
native_previous_base = cb1[5].y
```

and copy from:

```text
current:  native_current_base  + 3 + localBone * 3 + row
previous: native_previous_base + 3 + localBone * 3 + row
```

into the global capture store at `capture_store_base`.

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
run = CustomShader_ExtractCB1
x101 = <local_bone_count>
cs-t2 = ResourceLocalToGlobalBoneMap_<part>
run = CustomShader_GatherBones
vs-t0 = ResourceLocalFakeT0_SRV
run = CustomShader_RedirectCB1
vs-cb1 = ResourceFakeCB1
```

`RedirectCB1` is part of the default consume path. Once `ResourceLocalFakeT0_SRV` is bound as `vs-t0`, the original native `cb1[5].x/y` would point outside the local fake store. `RedirectCB1` must preserve all current `cb1` values except shader-visible `cb1[5].x/y`, which are rewritten to the local fake palette bases:

```text
cb1[5].x = 0
cb1[5].y = LOCAL_PREVIOUS_ROW_OFFSET
```

If a future design binds the full global capture store directly as `vs-t0`, then `cb1[5].x/y` must instead be redirected to that part's global current/previous offsets. Under the current small-buffer consume design, the correct redirect target is always `0/1024`.

## Performance Direction

Avoid per-pass duplicate capture. If multiple passes share the same native palette identity, capture once where possible and let later consume passes reuse the same global store rows.

The scanner should choose a capture draw that appears before all consuming draws that need it. If that cannot be guaranteed, the integration layer should move the consuming draw to a later host or split the affected part.

Future optimization should target reducing draw-time CS count:

- Deduplicate capture records by native palette identity.
- Skip consume work on untouched original draws.
- Keep local palettes as static buffers generated offline.
- Keep capture metadata in a single static buffer generated offline.
- Avoid runtime structural checks; all verification should happen during scan/export.

## Static Buffer Tables

Runtime data should be table-driven and generated offline.

### MainCaptureBoneMap

```text
Buffer/MainCaptureBoneMap.buf
uint MainCaptureBoneMap[]

header uint4:
  x = record_count
  y = pair_table_uint_base
  z = total_pair_count
  w = flags/reserved

record uint4:
  x = pair_base
  y = pair_count
  z = source_bone_count
  w = flags/reserved

pair uint2:
  x = source_local_bone
  y = target_global_bone
```

`x100` selects the record. `cs-t2` binds `ResourceMainCaptureBoneMap`. Main capture normally stores direct pairs such as `local 0 -> global_base + 0`, while LOD capture uses the same pair shape in a separate LOD file.

### LocalToGlobalBoneMap

```text
Buffer/<region>_partNN-LocalToGlobalBoneMap.buf
uint LocalToGlobalBoneMap[local_bone_count]

LocalToGlobalBoneMap[part_local_bone] = canonical_global_bone
```

Each consume command list binds the part's local-to-global map as `cs-t2`. `GatherBones` uses it to copy from the global capture store into the local fake `vs-t0`.

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
7. Apply Global Bone Pool when source mesh vertex groups should be converted/normalized
8. Analyze LOD mappings, if LOD FrameAnalysis folders are configured
9. Organize/edit meshes inside the same single root collection
10. Generate buffers, BoneMerge runtime tables, and INI
11. Inject/modify main INI, if enabled
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
replay parts: N
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

This root collection is a large working collection containing imported source objects and export
IB region subcollections. Export recursively walks mesh objects under valid region/part
subcollections in this root and writes buffers/INI from that structure.

The root contains IB region collections. Each region collection can export either one implicit
part or multiple explicit parts. This mirrors the `mod_importer` rule while keeping BoneMerge's
single-root workflow:

```text
ExportRoot
  <ib_hash>-<match_index_count>-<first_index>
    mesh objects...            ; implicit part00
  <ib_hash>-<match_index_count>-<first_index>
    part00
      mesh objects...
    part01
      mesh objects...
```

The region collection controls the output IB identity and match range. Part collections control
local palette boundaries and draw parts. Mesh object names are free-form during export. Object
names must not be used as the source bone mapping identity.

### Hash Collection Identity

Each export region collection name must clearly contain the output IB hash, `match_index_count`,
and `first_index`. The recommended name is:

```text
<ib_hash>-<match_index_count>-<first_index>
```

Examples:

```text
2e5d9294-23220-0
```

`match_index_count` is intentionally visible in the collection name so the exported draw identity
is clear in Blender. The scanner still verifies it against the current FrameAnalysis and the
latest capture manifest. Part collections are named `part00`, `part01`, etc. If an IB region uses
direct mesh objects, they are treated as one implicit `part00`. If a region mixes direct mesh
objects and explicit part collections, Prepare Export should either migrate direct meshes into
`part00` or stop with a clear diagnostic; the preferred UI behavior is automatic migration because
it preserves the user's visible result.

### Bone Localization

Before export, the addon may clean empty vertex groups. Then it builds the final part palette from
the numeric vertex groups actually used by visible objects inside each part.

The export-time assumption follows `mod_importer`:

```text
Blender vertex groups use global bone-number semantics before localization.
```

Export localizes per part:

```text
global bone -> part local bone
```

The exported blend/index buffer uses compact part-local bone indices starting at zero. The part
palette file maps those local indices back to canonical global capture-store bone indices. Vertex
group names are sorted numerically before palette assignment, so the mapping is stable and easy to
review:

```text
visible vertex groups in part: 0, 4, 9, 31
Blend indices written:         0, 1, 2, 3
Palette file written:          0, 4, 9, 31
```

Hard limit:

```text
part local bone count <= 256
```

This limit applies after recursive visible-object gathering and bone localization. If a region's
implicit part exceeds 256 bones, Prepare Export should auto-split into multiple parts:

```text
1. Try object-level splitting first, preserving whole objects when possible.
2. If one object exceeds 256 used bones, split that object by triangles and duplicate vertices as needed.
3. Every generated part receives its own palette and draw call.
```

The splitter must optimize for correctness before minimal part count. A stiff or slightly less
optimal split is acceptable; a part that writes a local index over 255 is not.

### Part Buffer Output

Every exported part writes a complete set of resources based on the region's `vertex_layout_key`:

```text
<region>_partNN-Position.buf
<region>_partNN-Texcoord.buf
<region>_partNN-Blend.buf
<region>_partNN-Index.buf
<region>_partNN-LocalToGlobalBoneMap.buf
```

The IB format is always:

```text
DXGI_FORMAT_R32_UINT
```

This avoids vertex-count limits and simplifies object/triangle splitting. The VB writers must use
the actual `vertex_layout_table` fields for the target region. The region collection decides the
export layout; individual mesh object metadata is not allowed to override or veto that layout. Do
not use one fixed layout template for all IBs.

### Import Into Root

Import Selected IBs places imported objects directly under the single root collection. It does not create a separate import root.

After Build Global Bone Pool, the addon creates empty child collections for every accepted candidate:

```text
<root export collection>
  <ib_hash>-<match_index_count>-0
  <ib_hash>-<match_index_count>-0-NO_CAPTURE_BONES
  <ib_hash>-<match_index_count>-0-NO_LOD_DYNAMIC_VB0
```

These child collections are the export contract. Final INI identity, `match_index_count`, palette generation, and delayed shadow replay all derive from the child collection name. Mesh object names are not final export identity. The `NO_CAPTURE_BONES` suffix marks mapping-only candidates: their own native matrices were not captured in the early shadow stage, but the collection can still host meshes that use other captured global bones.

`NO_LOD_DYNAMIC_VB0` marks candidates whose imported `vb0` position stream is not from the same backing buffer as the IB/static VB family. In the observed eyelash case `2009f0d6-1356-0`, `ib` is backed by `9a09f1f0` while `vb0` is backed by CPU-uploaded `1d6a6186`; this means the geometry is already preprocessed/pre-skinned for the pass and is not a stable spatial source for LOD matching. These entries may still be capture-ready in the main pool, but LOD matching and LOD coverage review must ignore their global slots explicitly.

Users manually move/edit/copy objects into the desired child collections for export.

The addon may provide a convenience button:

```text
Add Selected To Matching Export Collections
```

This button reads each selected object's stored source hash or first independent 8-hex source hash, finds a unique matching child collection under the root, and links the object there. This is only a workflow accelerator. If the same hash has multiple child collections with different `match_index_count`, the operation must report `ambiguous_source_hash` instead of guessing. The final export identity is still determined by the destination child collection.

### Candidate List And Rename

Global bone mapping is driven by a UI list, not by an extra Blender collection. Too many collections make the workflow harder to understand.

The Candidate IB list is the single editable source for import and global-pool building. Analyze fills it automatically from FrameAnalysis. Users can add/remove entries manually, and a refresh action can merge source hashes from objects currently under the root collection.

For collection refresh and compatibility renaming, the object source hash is the first independent 8-hex value in the object name or stored custom property. This is used only for refreshing/building/renaming the source mapping list, not for export identity.

The scanner uses the source hash to find current FrameAnalysis data:

```text
source IB hash
match_index_count
source_index_count
compact local bone count
used source-local bone indices
per-candidate skin weight/index format
runtime validation fields
```

Build Global Bone Pool creates:

```text
used source local bone -> compact global bone
capture records sorted by capture availability, then match_index_count descending
MainCaptureBoneMap / global pool manifest entries
the single export root collection, if missing
empty export child collections for accepted candidates, with unavailable markers for no-shadow mappings
```

Build Global Bone Pool is data-only. It must not rename vertex groups or merge seam groups. This keeps the pool construction repeatable and lets users rebuild the manifest/collections without mutating meshes.

Apply Global Bone Pool is the separate mutating step for imported/source objects. It renames matched source objects into global vertex-group semantics and then runs seam-group normalization. This is a convenience step for objects imported by this plugin, because they already carry `bmc_source_ib_hash`, `bmc_match_first_index`, and `bmc_match_index_count`.

Rename logic:

```text
object source hash + object local bone index -> global bone index
```

The rename must set object metadata:

```text
bmc_vertex_group_state = "global"
bmc_global_group_remap = [...]
bmc_global_source_key = "<ib_hash>-<match_index_count>-<match_first_index>"
bmc_global_pool_generation = "<manifest generation id>"
```

If an object already has matching `bmc_vertex_group_state = "global"` and the same generation/source key, repeat clicks must skip it instead of remapping global groups as if they were local groups.

Analyze must not silently rewrite meshes.

This rename function is an important compatibility feature. For external models imported by other tools, the plugin can still convert local numeric groups to the global pool naming scheme by using the object hash and the built source mapping table.

External/export objects use a separate button:

```text
Apply Global Names In Export Collection
```

This walks mesh objects under export child collections. It resolves the mapping from the child collection name, not from the object name:

```text
child collection: 640d1c0e-46845-0
mapping source:   global pool record 640d1c0e + 46845
operation:        local numeric vertex group -> canonical global vertex group
```

This makes the collection the authority for renamed external meshes. Object names can be free-form after the mesh is placed in a child collection.

There is a default-closed `Vertex Group Tools` panel for temporary or external meshes:

```text
Rename To Global
Rename To Local
Merge Seam Groups
```

The rename buttons do not use collection membership. They inspect selected mesh objects, or all scene mesh objects when nothing is selected, and look for any 8-hex IB hash in the object name. If the hash uniquely matches a global-pool record, `Rename To Global` applies that source-local to global mapping. `Rename To Local` applies the inverse mapping and clears the global rename metadata. If one hash maps to multiple pool records, the object name must contain the full `hash-index_count-first_index` source key; otherwise the operation must report ambiguity instead of guessing.

`Merge Seam Groups` reuses the same seam vertex-group matcher as the original standalone seam tool. It builds a seam-space mapping from selected mesh objects and renames matched seam groups to the canonical group. Apply Global Bone Pool also calls this same function once after renaming candidate source meshes to global groups, so imported IB pool meshes get seam-group normalization without using the manual tool. A seam failure must warn instead of cancelling the apply step.

The seam matcher is performance-sensitive. It must cache each mesh's weighted boundary vertices and spatial hash once, then reuse that cache for object-pair matching. Vertices without numeric weights are excluded from seam matching because they cannot contribute to a vertex-group alias. If two matched seam groups already have the same global group number, the matcher treats it as an idempotent no-op rather than an error.

Final Blender contract before Prepare Export:

```text
Object name hash: used for auto-placement only.
Child collection: authoritative export identity.
Visible vertex-group numbers: canonical global bone indices.
Prepare Export: localizes evaluated mesh data into part-local indices, writes per-part LocalToGlobalBoneMap files, and materializes the runtime capture/gather resources.
```

### Import/Export/INI Rules

Import, export, and base INI generation should follow `E:/vscode/mod_importer`.

Notable inherited rules:

```text
Analyze picks the resolved draw/model using mod_importer discovery logic.
Import should prefer the output/display path with the strongest o-slot evidence where that is the mod_importer rule.
Texture marking should reuse mod_importer candidate and material semantics.
Export should reuse mod_importer evaluated-mesh, shape-key, texture, and collection-tree ideas.
INI generation should reuse mod_importer fast-path syntax where possible.
```

BoneMerge differs from mod_importer in the final VB/IB writer. BoneMerge must write the true
FrameAnalysis vertex layout for each IB region, including packed normals and non-UV TEXCOORD
fields. It should reuse mod_importer algorithms where they match, but not force BoneMerge meshes
into mod_importer's fixed runtime buffer layout.

BoneMerge adds the global bone capture/replay tables, per-part canonical palettes, LOD scatter
capture, and delayed shadow replay scheduling on top of the shared workflow.

Texture overrides are the exception to direct INI reuse: candidate detection and texture conversion can reuse `mod_importer`, but replacement output uses `TextureOverride hash = <texture_hash>; this = Resource...` rather than slot-style `ps-t` assignment.

## Analyze And Import Flow

The UI should expose four core build buttons before export:

```text
Analyze
Import Selected IBs
Build Global Bone Pool
Apply Global Bone Pool
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

Analyze produces one manifest. Import, Build Global Bone Pool, Apply Global Bone Pool, Generate, and Inject should consume this manifest instead of rescanning FrameAnalysis independently.

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
  "source_local_bone_count": 83,
  "used_local_bone_indices": [0, 1, 4, 9],
  "skin_format": {
    "slot": "vb2",
    "stride": 12,
    "blend_weights_format": "R16G16B16A16_UNORM",
    "blend_weights_offset": 0,
    "blend_indices_format": "R8G8B8A8_UINT",
    "blend_indices_offset": 8
  },
  "texture_region_key": "ae1ab184-8148-190539",
  "import_paths": {
    "ib": "",
    "vb": {},
    "layout": ""
  }
}
```

`enabled` is user-editable. Disabled candidates stay in the manifest for diagnostics but are not imported and do not build the global pool.

`local_bone_count` is the compact count of source-local bones that are actually referenced by weighted vertices. `source_local_bone_count` is the original native palette span, normally `max(used_local_bone_indices) + 1` unless the native draw proves a larger valid range. `used_local_bone_indices` is the authoritative sparse local palette list for Blender renaming and compact global-pool construction.

`skin_format` is per candidate, not global. Import and compact-palette analysis must read weights and indices from this candidate's selected `import_draw_index` and its own skin slot. For example, one IB may use `R16G16B16A16_UNORM` weights plus `R8G8B8A8_UINT` indices, while another uses `R32G32B32A32_FLOAT` weights plus `R32G32B32A32_UINT` indices. Header/layout and `.buf` files from different passes must not be mixed when decoding skin data.

For `.buf` decoding, the header/deduped layout `byte offset` is authoritative. The inline `vertex-data` preview in a `.txt` dump is useful evidence, but it must not be used to auto-shift the buffer base. In the tested `640d1c0e-46845-0` dump, trusting the preview would shift `vb2` by `-4` bytes and produce a false `0..255` fragmented weight palette; trusting the layout offset gives a stable 145-bone palette with per-vertex weights summing to 1.

`shadow_capture_ready` means the candidate has an early-shadow draw that can provide native bone matrices. Candidates where this is false can still enter the global mapping pool, but their generated export collection must be visibly marked as bone-capture unavailable. They are useful as draw/material hosts, not as sources of captured bones.

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
  "source_local_bone_count": 83,
  "used_local_bone_indices": [0, 1, 4, 9],
  "capture_store_base": 378,
  "bone_capture_available": true,
  "lod_match_excluded": false,
  "lod_match_excluded_reason": ""
}
```

`bone_pool_order` is compact. It allocates one global slot per entry in `used_local_bone_indices`, not one slot per number in `0..max_local`. Capture-ready candidates are sorted first; mapping-only/no-shadow candidates are sorted after them and are marked `bone_capture_available=false`.

`bone_capture_available` and `lod_match_excluded` are separate concepts. A dynamic/pre-skinned `vb0` candidate can still be `bone_capture_available=true` for main runtime capture while `lod_match_excluded=true` prevents the LOD matcher from using its positions or requiring the LOD scatter pass to fill its global slots.

The global bone index and capture-store index have the same meaning in HLSL. Exported part palettes must reference these indices exactly. Runtime capture for sparse native palettes must use the same `used_local_bone_indices` table so local source bone `248` can be written to compact global slot `base + compact_index`, not blindly to `base + 248`.

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

The inline `pairs` array is for diagnostics and project portability. The generated runtime should materialize it into a compact static `LodCaptureBoneMap.buf` when LOD is enabled.

#### lod_mapping

`lod_mapping` is the diagnostic per-canonical-bone table generated by Analyze LOD. It is not the final runtime buffer by itself; `lod_capture_records[].scatter_pairs` is the grouped runtime input.

```json
{
  "canonical_global_bone": 45,
  "lod_record_key": "lodhash00-12345-0",
  "lod_local_bone": 12,
  "score": 18.5,
  "votes": 57,
  "average_distance": 0.002,
  "status": "matched"
}
```

The first implementation writes these LOD fields into the main capture manifest only. INI/HLSL generation should consume the same fields later, but Analyze LOD itself must not emit runtime files yet.

For performance, Analyze LOD must use a lightweight point-cloud reader that loads only IB, POSITION, BLENDWEIGHTS, and BLENDINDICES. It must not run the full Blender import decode path for UVs, normals, or custom attributes. Before nearest-neighbor voting, weighted point clouds are compressed with semantic-preserving spatial buckets so runtime-scale meshes do not require every source vertex to participate in matching.

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

#### Data Type Registry

Known shader input contracts and observed vertex-buffer profiles live under `core/data_types/`.
This folder is a data fact registry, not a preset system. `vs_input_contracts.json` records only
semantics confirmed from readable `*-vs_replace.txt` disassembly, while
`vertex_layout_profiles.json` records recurring physical slot layouts such as UV0 plus packed
auxiliary `R8G8B8A8_SNORM` bytes. FrameAnalysis remains the source of truth for every import and
export offset, stride, format, and index count; the registry only annotates manifest records so
diagnostics can distinguish "declared in IA layout" from "actually consumed by this VS".

When a VS disassembly is missing, the registry must leave the VS role unknown instead of guessing
from IA declarations. In particular, a layout may expose the same four bytes as both `TEXCOORD2`
and `TEXCOORD4`, but only the disassembled VS decides which semantic is meaningful for that pass.

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
stride/layout are per candidate import draw
+0   TEXCOORD0  R32G32_FLOAT       UV0
+8   TEXCOORD1  R32G32_FLOAT       UV1, when present
+16  TEXCOORD4  R8G8B8A8_SNORM     packed auxiliary data, when present at this offset
```

`UV1` is real source data when the selected import draw exposes `TEXCOORD1`. Import should create a second Blender UV layer for it, for example `UV1`. If `TEXCOORD1` is absent, import and export copy `UV0` into `UV1` as the compatibility fallback. Do not synthesize UV1 from another pass layout while reading the selected draw's `.buf`.

`TEXCOORD4` is not a UV layer. It is a packed signed-normalized auxiliary field. It is used by multiple passes under different semantic names, for example `TEXCOORD4` and sometimes `TEXCOORD2` at the same byte offset. Import should keep it as a custom packed attribute or raw byte payload, not as a Blender UV map.

`Blend` buffer:

```text
stride/slot/layout are per candidate import draw
+0   BLENDWEIGHTS0  R16G16B16A16_UNORM or R32G32B32A32_FLOAT
+8/+16 BLENDINDICES0 R8G8B8A8_UINT or R32G32B32A32_UINT
```

`BLENDINDICES0` are source-local palette indices. They are not global bone indices. Import must decode them using the current candidate's `skin_format`, never a project-wide default. Import should keep them as local vertex groups until Apply Global Bone Pool renames or maps them. Export localizes each final part again and writes compact local indices. The per-part `LocalToGlobalBoneMap.buf` records local index -> canonical global bone.

The source `IB` can be `DXGI_FORMAT_R16_UINT`, even if older exported mods wrote `DXGI_FORMAT_R32_UINT`. Import should record the native index format from FrameAnalysis for diagnostics, but BoneMerge export always writes `DXGI_FORMAT_R32_UINT`. This avoids vertex-count limits and keeps automatic part splitting simple.

#### Multi-Pass Layout Evidence

The importer must choose one `import_draw_index` for one candidate IB and bind every imported slot to that draw's header/layout and `.buf`. Other draw hits are evidence for pass scheduling, material marks, or diagnostics; they must not be merged into the imported skin layout because weight/index formats can differ between IBs.

For the tested `640d1c0e` group, the same core slices were reused across many hits:

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
Export UV1 from the second UV layer only when the target layout truly contains TEXCOORD1.
Export normals from Blender custom split normals, then octahedral-encode them for NORMAL0.
Keep packed non-UV auxiliary fields such as TEXCOORD4 as custom/raw attributes.
If a required UV/raw auxiliary attribute is missing on an edited mesh, use an explicitly documented profile fallback or stop with a diagnostic instead of writing garbage.
```

Shape key and morph handling should be reused from `E:/vscode/mod_importer` wherever possible. BoneMerge should not invent a separate shape-key interpretation. The user-facing rule remains: the mesh visible in Blender is the mesh that is written.

Export is the inverse of import:

```text
Blender-space position -> game-space POSITION0
Blender custom split normal -> game-space normal -> packed NORMAL0
Blender UV layers -> TEXCOORD0/TEXCOORD1 only when the target layout contains those fields
raw/custom packed auxiliary attributes -> TEXCOORD2/4/5/6 fields exactly as the target layout expects
global numeric vertex groups -> part-local BLENDINDICES0 via the part palette
normalized visible weights -> BLENDWEIGHTS0 in the target field format
Blender triangle winding -> game winding inverse of the import transform
```

The exporter must build the final vertex stream from evaluated loop data, not only Blender vertex
data. A new exported vertex is needed whenever any field that is written to the target layout
differs:

```text
position
normal
UV0/UV1
raw packed auxiliary values
top-four normalized weight/index tuple
```

This protects hard normals, UV seams, material seams, and triangle-level part splits. If the object
was imported with X mirror conversion, export applies the inverse mirror conversion before packing.
The per-object mirror setting must be taken from object metadata first, then the scene default; do
not guess from mesh coordinates.

### Export Buffer Writer Contract

Each region uses the actual `vertex_layout_table[region_key]` as the writer contract. For every
part, create a zero-filled bytearray of `vertex_count * stride` for each required VB slot, then
`pack_into` each field using the field's exact offset, format, and component width.

Required behavior:

```text
POSITION0 R32G32B32_FLOAT:
  write evaluated game-space position.

NORMAL0 R32_FLOAT:
  encode Blender custom split normal with the BoneMerge/game octahedral codec.

NORMAL0 R32G32B32_FLOAT:
  write normalized game-space float normal.

TEXCOORD0 R32G32_FLOAT:
  write primary UV layer.

TEXCOORD1 R32G32_FLOAT:
  write the second UV layer when present. If the target layout requires TEXCOORD1 and it is missing,
  stop export unless that layout profile explicitly permits a UV0-copy fallback.

TEXCOORD2/4/5/6 packed or float fields:
  write preserved custom/raw attributes. If missing, stop export unless a documented profile fallback exists.

BLENDWEIGHTS0:
  write top-four visible weights, normalized and encoded to the target format.

BLENDINDICES0:
  write part-local palette indices encoded to the target format. R8 indices require local index <= 255.
```

All non-UV TEXCOORD fields are data fields, not Blender UV maps. Their raw bytes must survive an
untouched import/export round trip.

### Build Global Bone Pool

Build Global Bone Pool consumes the editable Candidate IB list. There is no separate mapping collection or preset list in the simplified workflow.

The accepted pool inputs are:

```text
enabled candidate
local_bone_count > 0
```

Candidates with `shadow_capture_ready == true` are capture sources and must be placed first. Candidates without early-shadow capture are still allowed into the mapping pool after all capture-ready candidates. Their child export collections are created with a visible unavailable marker, because their own bones cannot be collected; they can still host delayed/material replay for meshes that use already-captured global bones.

Accepted candidates are sorted by capture availability, then descending `match_index_count`, with larger source parts occupying earlier global-bone positions and earlier vertex-group naming ranges. Ties use compact local bone count, mesh fingerprint, first index, and hash only as deterministic tie-breakers.

The pool is compact:

```text
global = global_bone_base + compact_index
compact_index = index of source_local_bone in used_local_bone_indices
```

This means imported raw vertex groups may be sparse and overlapping across IBs, but after Apply Global Bone Pool global-renames them they become compact and globally unique for the selected source identity. Runtime capture must therefore use the same sparse-source table; a continuous copy of `0..source_local_bone_count` is no longer semantically correct for compact pools.

It also creates the single export root collection if it does not exist. After this step, users edit models and place final meshes under IB region collections inside the export root.

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
canonical global pool built from enabled candidates, with capture-ready records first
canonical global bone indices
final export region parts and LocalToGlobalBoneMap semantics
```

LOD does not change exported VB/IB resources. Exported region parts, their R32 IBs, and their
part-local palettes are canonical high-detail replacement resources. Main and LOD runtime paths
only differ in how they fill the canonical global bone pool before replay:

```text
main path:
  native main local bone -> canonical global bone

LOD path:
  native LOD local bone -> one or more canonical global bones
```

Therefore a part palette always maps:

```text
part local bone -> canonical global bone
```

It never maps to an LOD-local bone. When LOD is active, LOD scatter capture must fill every
canonical global bone used by visible exported parts, or export/generation must report the missing
global bones.

LOD FrameAnalysis folders are optional secondary sources. They are analyzed only after Build Global Bone Pool, because LOD mapping needs the canonical global bone indices created from the user-confirmed Candidate IB list.

### LOD Flow

The intended user flow is:

```text
1. Analyze Main.
2. Import selected main candidates.
3. User deletes unwanted candidates from the Candidate IB list or refreshes the list from collection objects.
4. Build Global Bone Pool from enabled candidates; capture-ready entries are first, mapping-only entries are marked unavailable, and dynamic/pre-skinned `vb0` entries are marked as no-LOD sources.
5. Analyze LOD.
6. Generate runtime buffers and INI.
```

This keeps one editable list as the UI source of truth:

```text
Candidate IB List:
  all main candidates discovered from FrameAnalysis, plus manual or collection-refreshed entries
  enabled entries control import
  enabled + local_bone_count > 0 entries control Build Global Bone Pool
```

The LOD analyzer must consume the built global pool, not the full Candidate IB list. Imported candidates that the user deleted or disabled must not pollute LOD matching. Mapping-only/no-shadow candidates can appear in the pool, but they must not be treated as native capture sources unless an LOD mapping explicitly provides a valid capture path. Dynamic/pre-skinned `vb0` candidates are a special ignored class: their collection and pool record are marked, their geometry is skipped when building the canonical/LOD point clouds, and their global slots are not counted as required missing bones during LOD review.

### LOD Repair Fallback

LOD Repair is an advanced, default-collapsed tool. It is not part of the normal exact matching path and must be presented as a cautious fallback.

Prepare Export must check the final export collection against the latest `lod_mapping`:

```text
if unmatched global bones are not used by exported meshes:
  export normally

if unmatched global bones are used by exported meshes:
  block export and ask the user to use LOD Repair
```

The fallback tool works from the exported mesh weights, not from the original LOD mesh. For each used unmatched global bone `G`, it searches for a matched donor global bone `H`:

```text
1. Prefer matched groups that share vertices with G.
2. If none, choose a matched group on the same object with closest weighted point-cloud support.
3. If none, choose the closest matched group in the export collection.
4. If no donor exists, keep G unresolved and continue blocking LOD export.
```

Applying a fallback records:

```text
G -> inherit donor H
lod_record_key = H.lod_record_key
lod_local_bone = H.lod_local_bone
status = fallback_inherited
fallback_method
fallback_confidence
```

This may make the affected area stiffer, but it should avoid the destructive case where an unfilled or wrong global matrix twists the mesh. The UI must make this distinction clear: fallback means "safe inherited motion", not "true LOD bone match".

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
Buffer/LodCaptureBoneMap.buf
uint LodCaptureBoneMap[]

header uint4:
  x = record_count
  y = pair_table_uint_base
  z = total_pair_count
  w = flags/reserved

record uint4:
  x = pair_base
  y = pair_count
  z = lod_local_bone_count
  w = flags/reserved

pair uint2:
  x = lod_local_bone
  y = canonical_global_bone
```

Main capture remains linear or manifest-ordered:

```text
main local bone -> canonical global bone
```

LOD capture uses scatter:

```text
for pair in LodCaptureBoneMap[record.pair_base : record.pair_base + record.pair_count]:
    global_current[pair.canonical_global_bone] =
        native_lod_t0_current[pair.lod_local_bone]

    global_previous[pair.canonical_global_bone] =
        native_lod_t0_previous[pair.lod_local_bone]
```

The draw/consume path stays unchanged:

```text
part local bone -> LocalToGlobalBoneMap -> canonical global bone -> local fake vs-t0
```

This is the key LOD contract: final replacement geometry still uses the high-detail canonical global bone semantics. LOD only changes which native source matrices fill those global slots before drawing.

Generated replay with LOD still draws the same exported region parts:

```text
Gather part00 palette -> bind local fake vs-t0 -> draw part00
Gather part01 palette -> bind local fake vs-t0 -> draw part01
...
```

The gather step is identical for main and LOD because both paths have already populated the same
canonical global store. The only LOD-specific buffers are the scatter capture metadata/pairs used
before replay.

### LOD TextureOverrides And INI Output

Generated INI must include `TextureOverride` sections for resolved LOD IB hashes and `match_index_count` values. The logic is the same shape as the main path, but the capture shader and metadata are LOD scatter-aware.

For each resolved LOD source IB:

```text
TextureOverride LOD capture:
  hash = lod_ib_hash
  match_index_count = lod_match_index_count
  if vs == expected_lod_capture_vs:
      bind LodCaptureBoneMap
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
      draw the high-detail replacement parts that target this LOD stage
```

Unmodified LOD source IBs are left on the game's original path. Modified/exported LOD source IBs follow the same delayed shadow replay rule as main parts:

```text
1. Capture all required LOD native palettes into the canonical global pool.
2. Skip only original LOD shadow draws whose source IBs are replaced.
3. Leave unmodified LOD shadow draws untouched.
4. At the final compatible LOD shadow host, replay all delayed modified replacement parts.
5. Draw transparent shadow batches first, then bind white shadow PS resources and draw normal shadow batches.
```

The replay host and resource bindings are LOD-stage-specific, but the replacement part resources and part-local `LocalToGlobalBoneMap` files still use canonical high-detail global bone semantics.

## Replay Scheduling

The core replay goal is that any exported part can use bones captured from any source IB. A modified IB may reference bones from draws that occur later than its original shadow draw. Drawing that modified IB at its original shadow position can therefore read incomplete bone data.

To avoid this, modified shadow drawing is delayed:

```text
1. Early/source encounters capture native bone palettes.
2. Original shadow draws for modified/exported IBs are skipped at their original positions.
3. Unmodified game IBs are left alone and keep their original shadow drawing.
4. At the final compatible shadow host, all delayed modified shadow parts are replayed after the global bone store is complete.
```

The replay host is not limited to drawing only its own IB. It is a late scheduling point that can draw replacement parts from multiple source IBs, because those parts now gather from the completed global bone store.

Replay host selection:

```text
host_draw_index > max(required_capture_draw_index)
host is the last matching draw in the compatible render-state group
host pass has render/depth/blend/texture state that replacement parts can inherit
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
1. replay transparent shadow parts first, without binding white PS resources
2. bind the white shadow PS resource once
3. replay normal/opaque shadow parts
```

Normal/opaque parts need the white PS resource because they are being delayed from the earlier normal shadow draw into the later transparent host. Transparent parts inherit the transparent shadow state and do not bind white.

The host can be any compatible late draw in the cluster. It does not have to be the same IB as every replayed part. If the host IB itself is exported and skipped, the override still acts as the scheduling carrier; if the host IB is unmodified, its original draw should remain and the replay draws are appended by the generated override logic.

When a replay host draws multiple parts with one reusable local bone buffer, each part must gather and draw immediately:

```text
Gather part A -> bind local vs-t0 -> draw part A
Gather part B -> bind local vs-t0 -> draw part B
```

Do not gather several parts first and draw later, because the local buffer is overwritten by each gather call.

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
Analyze Main -> Candidate IB List -> Import Selected -> Build Global Bone Pool -> Apply Global Bone Pool -> Export -> Generate INI
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
  prune empty groups, sort numeric global groups, localize per part from zero, enforce <= 256 local groups per part, and auto-split oversized regions before writing part palettes.

Raw auxiliary attributes:
  preserve required packed fields such as TEXCOORD4, or stop export with a clear diagnostic when an edited mesh lacks them.

R32 index buffers:
  export all replacement IBs as DXGI_FORMAT_R32_UINT and verify generated drawindexed ranges.

WYSIWYG loop export:
  export evaluated loop data so custom normals, UV seams, and split triangles survive.

LOD:
  keep optional for the UI, but generation must check exported global bones against LOD coverage when LOD mappings are present.
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
   one root collection, IB region subcollections, optional part subcollections, WYSIWYG evaluated loop export.
6. Export buffer package:
   R32 IB, true-layout VB writers, per-part palette files, debug export manifest, auto-split for >256 local bones.
7. Static runtime buffers:
   MainCaptureBoneMap, LocalToGlobalBoneMap, optional LodCaptureBoneMap, global/local current and previous matrix stores.
8. INI generation:
   capture overrides, gather/replay overrides, texture hash overrides, delayed shadow replay plan.
9. LOD analysis and scatter capture:
   add after the main non-LOD loop can import, export, and replay correctly.
```

The first usable milestone should be:

```text
Analyze Main -> Import Selected -> Build Global Pool -> Export one edited IB region collection -> generated INI renders without deformation.
```

The first export-only milestone should come before INI generation:

```text
Import one untouched source IB -> place it in its matching region collection -> Export Buffers
-> compare debug manifest against vertex_layout_table
-> verify R32 IB, VB strides, field offsets, packed normals, raw TEXCOORD fields, and part palette.
```
