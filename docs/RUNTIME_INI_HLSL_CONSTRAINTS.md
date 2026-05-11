# Bone Merge Runtime INI/HLSL Constraints

This document records the current runtime contract for Bone Merge Capture. It is
the source of truth for the next INI/HLSL rewrite, including the current
multi-instance contract.

## Scope

The runtime does only two jobs:

1. Capture native game bone matrices into a canonical global bone pool.
2. Gather the bones needed by an exported part into a small local `vs-t0`, then
   draw replacement geometry under the game's current render state.

Import/export, texture marking, mesh splitting, and palette construction are
offline Blender responsibilities. Runtime shaders must not rediscover structure
that the Blender exporter can validate and write into static buffers.

## Export Collection Contract

The Blender export collection tree is the only source of truth for exported
geometry ownership. Mesh object names are descriptive metadata only; they must
not decide which IB/part owns a buffer.

Supported layouts:

```text
BMC Export Sources
  <ib_hash>-<match_index_count>-<first_index>
    direct mesh objects...           -> implicit part00

BMC Export Sources
  <ib_hash>-<match_index_count>-<first_index>
    part00
      mesh objects...                -> explicit part00
    part01
      mesh objects...                -> explicit part01
```

If a region has no `partNN` child collections, direct mesh objects on that
region are exported together as implicit `part00`. A single part produces one
replacement buffer set and one palette. Mesh objects inside that part are still
recorded as separate index ranges, so runtime emits one `drawindexedinstanced`
per mesh object while reusing the same IB/VB/palette resources.

If a region has any `partNN` child collection, only meshes recursively under
those explicit part collections are exported. Direct mesh objects on the region
collection are invalid in this mode and the exporter must stop with an error
that asks the user to move them into an explicit `part00`.

Child collections under an IB region are structural. If they contain meshes,
they must be named `partNN`; arbitrary nested child collections must not be
silently swept into implicit `part00`.

Only explicit `partNN` boundaries create additional buffer files and
`PartLocalToGlobalBoneMap` palettes. Multiple mesh objects in one part do not.

Prepare Export must not create, link, unlink, or move Blender collections or
objects. Auto-splitting over-large parts may create virtual part records and
buffer names, but it must not mutate the source collection tree.

Generated `geometry_buffers[*].object_names` must come from the freshly built
export plan for that buffer. Runtime regeneration must not backfill missing
object names from stale manifest `objects` records; stale metadata is worse
than no metadata because it can make the INI appear to draw objects that were
not exported.

## Namespace And Multi-Mod Isolation

One generated `BoneStore.ini` owns one Bone Merge runtime resource set.

The generated runtime should stay in one INI. Do not split the runtime into
multiple generated INIs for main capture, LOD capture, replay, or patch/fix
logic. HLSL can and should be kept in separate shader files, but all generated
3Dmigoto resources, overrides, draw orchestration, and frame reset hooks live in
the single generated `BoneStore.ini`.

Do not emit an explicit `namespace = ...` line. 3Dmigoto keeps resources from
separate loaded INIs isolated for this use case, and an explicit namespace adds
noise without helping the generated runtime.

Two separate generated INIs may reuse short resource names such as:

```ini
[ResourceGlobalBonePool_UAV]
[ResourceLocalBonePool_UAV]
[ResourceDumpedCB1_UAV]
[ResourceMainCaptureBoneMap]
```

They do not collide in normal generated output because the generator does not
opt into a shared explicit namespace.

`ShaderOverride` and `TextureOverride` hash matching still participates in the
global 3Dmigoto override system. Resource isolation does not prevent two mods
from matching the same game hash. Generated overrides therefore still need
specific `hash`, `match_index_count`, `match_first_index` where available, and
careful priority/duplicate behavior.

## ShaderOverride Contract

Only VS hashes proven to belong to the analyzed early shadow/capture window are
marked. The analyzer still records the primary normal/transparent VS pair for
replay ordering, but the capture filter can include extra VS hashes from the
same shadow window when those draws have skinned IB input and `vs-t0` bone data
available. This covers transparent tail/hair style parts whose shadow pass uses
a third VS.

```ini
[ShaderOverride_BMC_ShadowVS_A]
hash = <normal_or_transparent_shadow_vs_hash_a>
filter_index = 200
allow_duplicate_hash = overrule

[ShaderOverride_BMC_ShadowVS_B]
hash = <normal_or_transparent_shadow_vs_hash_b>
filter_index = 200
allow_duplicate_hash = overrule

[ShaderOverride_BMC_ShadowVS_Extra]
hash = <extra_capture_ready_shadow_window_vs_hash>
filter_index = 200
allow_duplicate_hash = overrule
```

These `filter_index = 200` entries are generated from FrameAnalysis/manifest
data, not from a static hand-maintained table.
If the analyzer records explicit shader filter rules with both `hash`/`vs_hash`
and `filter_index`, those manifest rules are also emitted before editable JSON
fallback rules.

`filter_index = 200` means only:

```text
this draw belongs to the early shadow/capture stage
```

It must not encode normal shadow versus transparent shadow. Those are separated
offline by the replay plan and by the IB/collection classification.

Do not generate broad VS override tables for visible, outline, material, or
effect passes outside the verified early shadow window. Visible replacement must
inherit the game's current pass state through `TextureOverride` branches.

Narrow runtime exclusions that are not part of capture, such as known residual
/ afterimage VS passes, live in `core/data_types/runtime_shader_filters.json`.
Those rules are merged after the FrameAnalysis-derived rules. The default
residual rule writes `filter_index = 204` and is controlled by the exporter UI
checkbox `过滤残影`.

## TextureOverride Branch Contract

Main and LOD overrides follow the same branch rules. The only difference is the
capture palette source: main records use `ResourceMainCaptureBoneMap`; LOD
records use `ResourceLodCaptureBoneMap` and scatter into the same canonical
global pool.

Each relevant IB override should have only two runtime branches:

```ini
if vs == 200
  ; shadow/capture branch. Capture bones here. Skip only when this IB has
  ; exported replacement geometry; delayed shadow replay happens at the final
  ; shadow draw, not at every source draw.
endif

if vs != 200
  ; visible/outline/material/effect replacement path under current game state.
endif
```

When residual filtering is enabled, the visible branch excludes the additional
filter index:

```ini
if vs != 200 && vs != 204
  ; normal replay path, but not residual/afterimage passes.
endif
```

Visible branches may also run `CustomShader_RecordBones` before replay. This is
the fallback for frames where the game does not issue the usual early shadow
capture draws before visible/effect draws. In that case, each matching visible
IB refreshes its own main/LOD capture record from the currently bound `vs-t0`
and `cb1[5].xy`; only branches with exported geometry then skip and replay.

Avoid many pass-specific `if` branches. The INI should be table-driven by
static buffers and by export/replay lists.

`handling = skip` is generated only when the export collection actually contains
replacement geometry for that source/host context. Capturing an IB does not by
itself imply skipping it.

`draw = from_caller` is not a general safety default. It is generated only on
the final shadow scheduling draw, and only when that final IB has no exported
replacement geometry of its own. This preserves the final native shadow draw
before appending delayed replacement shadow parts.

## Shadow/LOD Failure Postmortem And Hard INI Rules

The 2026-05-10 `ply` shadow regression had two separate causes:

```text
main menu shadow:
  a non-exported final shadow host was used for delayed replay
  but the host's own native draw was not preserved
  result: the host shadow, for example hair shadow, disappeared

LOD shadow:
  LOD shadow source draws were skipped and immediately replayed main exported
  parts from a single partial LOD capture record
  result: the original LOD shadow disappeared, and the replacement shadow could
  twist because the global pool did not yet contain every required canonical bone
```

These are INI-generation bugs, not mesh-buffer bugs.

Main and LOD use the same shadow replay contract. Their capture palettes differ,
but the skip/replay decision tree does not.

The hard rule is:

```text
For every main or LOD IB:
  if the export collection has no mesh objects for this IB:
      capture bones only
      do not skip
      do not replay replacement geometry for this IB

  if the export collection has mesh objects for this IB:
      in that IB's shadow branch, if vs == 200:
          handling = skip
      do not draw its shadow replacement immediately

At the final shadow draw for the character/shadow cluster:
  if the final host IB itself has no exported mesh objects:
      draw = from_caller
      then replay all delayed exported shadow parts

  if the final host IB itself has exported mesh objects:
      handling = skip
      then replay the host's replacement plus all delayed exported shadow parts

Visible/material/effect stages:
  replace vs-t0/cb resources and draw normally in the visible branch
```

Capture can stay outside the `if` branch because `RecordBones` validates the
current `cb1` window and `vs-t0` data before writing:

```ini
[TextureOverride_BMC_<source>]
hash = <source_ib_hash>
match_index_count = <source_index_count>
run = CustomShader_ExtractCB1
x100 = <record_index>
cs-t2 = ResourceMainCaptureBoneMap ; or ResourceLodCaptureBoneMap
run = CustomShader_RecordBones
```

But draw suppression and replay must remain branch-specific.

Capture-only main or LOD source:

```ini
[TextureOverride_BMC_<capture_only_source>]
hash = <ib_hash>
match_index_count = <index_count>
run = CustomShader_ExtractCB1
x100 = <record_index>
cs-t2 = ResourceMainCaptureBoneMap ; or ResourceLodCaptureBoneMap
run = CustomShader_RecordBones

; No handling = skip.
; No delayed replay here.
```

Visible replacement source with exported geometry:

```ini
if vs != 200 && vs != 204
  handling = skip
  run = CustomShader_ExtractCB1
  ; replay <exported_part>
  x101 = <part_local_bone_count>
  cs-t2 = ResourcePartLocalToGlobalBoneMap_<exported_part>
  run = CustomShader_GatherLocalBones
  vs-t0 = ResourceLocalBonePool_SRV
  run = CustomShader_RedirectCB1
  vs-cb1 = ResourceFakeCB1
  ib = ResourcePart_<exported_part>_Index
  vb0 = ResourcePart_<exported_part>_Position
  vb1 = ResourcePart_<exported_part>_Texcoord
  vb2 = ResourcePart_<exported_part>_Blend
  vb3 = ResourcePart_<exported_part>_Position
  drawindexedinstanced = <index_count>,INSTANCE_COUNT,0,0,FIRST_INSTANCE
endif
```

Early shadow source whose own original shadow is replaced later:

```ini
if vs == 200
  handling = skip
endif
```

Final shadow host that is not itself exported must preserve exactly that native
final draw before drawing delayed replacement parts:

```ini
[TextureOverride_BMC_<final_shadow_host>]
hash = <host_ib_hash>
match_index_count = <host_index_count>
run = CustomShader_ExtractCB1
x100 = <host_record_index>
cs-t2 = ResourceMainCaptureBoneMap
run = CustomShader_RecordBones

if vs == 200
  draw = from_caller
  ; delayed transparent shadow replay, when present
  ; replay <transparent_part>
  ...
  ps-t0 = ResourceBMCWhiteShadow
  ; delayed normal shadow replay
  ; replay <normal_part>
  ...
endif
```

Final shadow host that is itself exported does not preserve the native draw. It
must skip and replay its own equivalent replacement as part of the delayed batch:

```ini
if vs == 200
  handling = skip
  ; replay <host_equivalent_part>
  ; replay <other_delayed_parts>
endif
```

The known bad LOD pattern is:

```ini
[TextureOverride_BMC_<lod_source>_LOD]
hash = <lod_ib_hash>
match_index_count = <lod_index_count>
run = CustomShader_ExtractCB1
x100 = <one_lod_record>
cs-t2 = ResourceLodCaptureBoneMap
run = CustomShader_RecordBones

if vs == 200
  handling = skip
  ; BAD unless this single record already covers every required canonical bone.
  ; replay <main_exported_part>
endif
```

Generated LOD shadow replay must use a coverage proof:

```text
For every delayed replay part:
  required_globals = all values in PartLocalToGlobalBoneMap_<part>
  available_globals = union of all LodCaptureBoneMap records that are guaranteed
                      to have executed before the selected LOD shadow host

The LOD shadow host is valid only when:
  required_globals is a subset of available_globals
```

If that proof fails, the generated INI must not skip the LOD shadow draw. Leave
the native LOD shadow alone and generate capture-only LOD entries:

```ini
[TextureOverride_BMC_<lod_capture_only>_LOD]
hash = <lod_ib_hash>
match_index_count = <lod_index_count>
run = CustomShader_ExtractCB1
x100 = <lod_record_index>
cs-t2 = ResourceLodCaptureBoneMap
run = CustomShader_RecordBones

; No if vs == 200 handling = skip.
; No LOD shadow replay from this incomplete record.
```

When the proof succeeds, the LOD shadow replay is scheduled on the final LOD
shadow host recorded by the analyzed LOD shadow stage. Earlier LOD entries are
capture-only. The host branch is:

```ini
[TextureOverride_BMC_<verified_lod_shadow_host>_LOD]
hash = <lod_host_ib_hash>
match_index_count = <lod_host_index_count>
; Optional capture lines appear here only if this final host also has a LOD
; capture record. Scheduling replay on the final host does not require inventing
; a capture record for it.

if vs == 200
  ; Use draw = from_caller only when this host is not itself replaced.
  draw = from_caller
  ps-t0 = ResourceBMCWhiteShadow
  ; delayed normal shadow replay
  ; replay <main_exported_part_using_complete_global_pool>
  ...
endif
```

Or, if the verified host itself is one of the skipped LOD sources:

```ini
if vs == 200
  handling = skip
  ps-t0 = ResourceBMCWhiteShadow
  ; delayed normal shadow replay, including the host equivalent part
  ...
endif
```

Do not generate immediate LOD shadow replay from every LOD source. LOD capture
records often scatter different subsets of the same canonical skeleton; one LOD
draw can legitimately need bones captured by other LOD draws.

Do not generate `draw = from_caller` on non-final shadow/capture entries. A
capture-only entry without exported geometry should simply capture and leave the
game draw alone. An exported entry should skip its original shadow and wait for
the final shadow replay host.

## CommandList Usage Contract

Default generated INI should be readable in execution order. Prefer inline
logic inside the relevant `TextureOverride` branch unless a `CommandList` has a
clear reuse or lifecycle purpose.

Always allowed:

```ini
[Present]
run = CommandList_BMC_FrameEndReset
```

Frame reset is a lifecycle hook, so keeping it as a named command list is
acceptable even if it has only one caller.

Allowed only when reused by multiple branches or stages:

```text
shared draw block used by visible replay and shadow replay
shared draw block used by main and LOD replay
shared setup used by several TextureOverrides
```

Do not generate one-off wrapper command lists for simple capture or replay
steps. These should stay inline:

```ini
run = CustomShader_ExtractCB1
x100 = <capture_record_index>
; bind/select capture map
run = CustomShader_RecordBones

if vs != 200
  ; bind part map
  run = CustomShader_GatherLocalBones
  run = CustomShader_RedirectCB1
  ; bind vb0/vb1/vb2/vb3/ib
  drawindexedinstanced = ...
endif
```

The rule is:

```text
CustomShader = reusable algorithm
CommandList  = reused orchestration or frame lifecycle hook
Inline       = default TextureOverride control flow
```

Capture does not need to be guarded by `if vs == 200`. The compute shader must
validate the extracted `cb1[5]` bone window and `vs-t0` bounds before it writes
anything. Draw suppression and replay are still guarded separately, because
capturing an IB does not by itself mean the original draw should be skipped.

## Capture Record Selector

The runtime keeps one compact static capture table:

```text
MainCaptureBoneMap.buf
LodCaptureBoneMap.buf, when LOD is enabled
```

`x100` is allowed as a lightweight record selector:

```ini
x100 = <capture_record_index>
run = CustomShader_RecordBones
```

`x100` does not mean "where to store bones". Storage rules are entirely in the
static capture map buffer. `x100` means only:

```text
use record N from the currently bound capture map
```

The alternative, one capture-map buffer per IB, was rejected because it creates
too many resources and makes the INI noisier.

When the same override key can be executed in both main and LOD capture
profiles, the INI may emit a `$bmc_profile_lod` global:

```ini
[Constants]
global $bmc_profile_lod = 0
```

The recognized LOD chain start sets this flag under `if vs == 200`, the shared
record block branches between `ResourceLodCaptureBoneMap` and
`ResourceMainCaptureBoneMap`, and the LOD chain host plus frame lifecycle reset
the flag to `0`. This flag only selects the capture map; replay still uses the
part's `PartLocalToGlobalBoneMap`.

`x102` is reserved for future capture sequence flags. It must not be emitted or
read while the generator cannot prove the multi-instance capture sequence:

```text
bit 0 = begin capture sequence
bit 1 = end capture sequence
```

The current generated `RecordBones` shader does not read `x102`; it always uses
slot 0. Multi-instance generation should reintroduce begin/end only after the
analyzer has verified that the capture sequence order matches replay instance
slots.

## Capture Map Buffer Contract

`MainCaptureBoneMap` and `LodCaptureBoneMap` use the same semantic shape.

```text
uint4 header:
  x = record_count
  y = pair_table_uint_base
  z = total_pair_count
  w = flags/reserved

uint4 record[record_count]:
  x = pair_base
  y = pair_count
  z = source_local_bone_count
  w = flags/reserved

uint2 pair[total_pair_count]:
  x = source_local_bone
  y = target_canonical_global_bone
```

Main capture usually maps a source local palette directly into a compact global
range. LOD capture uses the same pair shape, but `source_local_bone` is a LOD
local bone and `target_canonical_global_bone` can be one-to-many across LOD
records.

The compute shader reads the current draw's native `cb1` slot and native
`vs-t0`. Game VS code addresses per-instance cb1 records as:

```text
instance_cb_base = SV_InstanceID * 16
native_current_base  = cb1[instance_cb_base + 5].x
native_previous_base = cb1[instance_cb_base + 5].y

current  row = native_current_base  + 3 + source_local_bone * 3 + row
previous row = native_previous_base + 3 + source_local_bone * 3 + row
```

For the observed capture stage, each shadow draw is single-instance, so the
native instance slot read by capture is normally slot 0. The destination
`capture_slot` is the runtime character instance slot assigned by the capture
sequence. Future instanced capture draws can use the same formula for multiple
native instance slots, but that is not the first target path.

Capture writes into the canonical global pool using the pair target and the
assigned `capture_slot`.

## Part Local-To-Global Map Contract

Every exported region part has its own part-local map:

```text
<region>_partNN-PartLocalToGlobalBoneMap.buf
```

Semantic:

```text
part_local_bone -> canonical_global_bone
```

All meshes inside the same `partNN` collection share this map. A second part
exists only when the region needs another <=256 local palette window.

The current flat map is valid when the INI passes the local count separately.
The target rewrite should prefer a tiny header so `GatherLocalBones` can read
the local count from the buffer and avoid another INI parameter.

## Shared HLSL Modules

Keep HLSL small and reusable:

```text
bone_store_common.hlsli
extract_cb1_vs.hlsl
extract_cb1_ps.hlsl
record_bones_cs.hlsl
gather_local_bones_cs.hlsl
redirect_cb1_cs.hlsl
reset_runtime_state_cs.hlsl
```

`record_bones_cs.hlsl` is shared by main and LOD capture. The bound capture map
decides whether the source is main or LOD.

`gather_local_bones_cs.hlsl` is shared by every exported part. The bound
`PartLocalToGlobalBoneMap` decides which canonical global bones to gather.

`redirect_cb1_cs.hlsl` preserves the current `cb1` content except the
shader-visible bone windows used by supported replay instance slots:

```text
for slot in supported replay slots:
  cb1[slot * 16 + 5].x = LocalCurrentBase(slot)
  cb1[slot * 16 + 5].y = LocalPreviousBase(slot)
```

This is required because consume binds a small local fake `vs-t0`; native
`cb1[slot * 16 + 5].xy` points into the game's large native store and is
invalid for the local buffer.

## Runtime Buffers

The generated runtime needs these mutable resources:

```text
DumpedCB1_UAV/SRV        extracted current draw cb1
FakeCB1_UAV/SRV          redirected cb1 for replacement draw
GlobalBonePool_UAV/SRV   canonical current and previous matrices per instance slot
LocalBonePool_UAV/SRV    per-part local current and previous matrices per instance slot
RuntimeState_UAV/SRV     counters, slot validity, frame/generation flags
```

The global and local bone pools both store current and previous matrices. They
have an explicit instance dimension:

```text
GlobalBonePool[instance_slot][canonical_global_bone][current/previous]
LocalBonePool[instance_slot][part_local_bone][current/previous]
```

The local fake `vs-t0` layout must preserve the game VS semantic offset of
`+3` matrix rows:

```text
BONE_ROWS = 3
CB1_INSTANCE_STRIDE = 16
LOCAL_CURRENT_ROW_OFFSET(slot)  = slot * LOCAL_SLOT_ROW_STRIDE
LOCAL_PREVIOUS_ROW_OFFSET(slot) = slot * LOCAL_SLOT_ROW_STRIDE + 1024

current  row = LOCAL_CURRENT_ROW_OFFSET(slot)  + 3 + local_bone * 3 + row
previous row = LOCAL_PREVIOUS_ROW_OFFSET(slot) + 3 + local_bone * 3 + row
```

`LOCAL_SLOT_ROW_STRIDE` must be large enough for both current and previous
windows. With the current fake local limit, `2048` rows is the default stride
for one instance slot.

Offsets are rows, while capture/store bases and palette values are bones.

## Geometry Binding Contract

Exported geometry uses collection/part layout, not per-object guesses.

Binding names:

```text
vb0 = Position
vb1 = Texcoord
vb2 = Blend
vb3 = Position
ib  = Index, DXGI_FORMAT_R32_UINT
```

`vb3` deliberately reuses the same resource as `vb0`.

Each draw command gathers local bones immediately before drawing that part:

```text
Gather part A -> bind local vs-t0 -> redirect cb1 -> draw part A
Gather part B -> bind local vs-t0 -> redirect cb1 -> draw part B
```

Do not gather multiple parts first and draw later, because the local pool is
overwritten by each gather.

## Shadow Replay Contract

Modified shadow drawing is delayed to the final compatible shadow host.

Replay order:

1. Draw exported transparent shadow parts first, without binding the white
   shadow PS resource.
2. Bind the white shadow PS resource once.
3. Draw exported normal/opaque shadow parts.

Transparent versus opaque membership is generated offline from the analyzed IB
/ collection / replay plan, not from separate `filter_index` values.

If one exported IB has both transparent and normal shadow hits in the analyzed
shadow stage, it belongs to both replay batches. Do not collapse it to only the
latest pass role, because that can drop one of the native shadow passes.

The final host can draw replacement parts from other IBs because the global
bone pool is complete by that point.

For LOD shadow replay, the host must be the actual final LOD shadow host from
the recognized LOD capture chain, not a global singleton and not a later
composite draw. A frame may contain several LOD capture chains; runtime export
may therefore emit several LOD shadow replay plans, each attached to its own
host key. The coverage proof in `Shadow/LOD Failure Postmortem And Hard INI
Rules` must still succeed before any LOD shadow skip/replay is emitted. The LOD
host uses the same final-shadow rule as main: if the final LOD host has no
exported geometry, emit one `draw = from_caller` before replay; if the final LOD
host has exported geometry, skip it and replay its equivalent replacement in the
delayed batch. Earlier LOD capture entries must not use `draw = from_caller`.

## Visible Pass Contract

For `vs != 200`, the replacement path should inherit the current game pass
state:

```text
main shading
outline
material/effect passes
```

The generated INI should bind replacement buffers and draw under the current
state, instead of maintaining a broad table of visible VS shaders.

Texture replacement uses texture-hash style overrides:

```ini
[TextureOverride_<semantic>_<texture_hash>]
hash = <texture_hash>
this = Resource<MarkedTexture>
```

Do not generate slot-style texture replacement as the Bone Merge-specific final
form.

## Multi-Instance Runtime Contract

The runtime supports same-character multi-instance by adding an instance slot
dimension to both bone pools:

```text
GlobalBonePool[instance_slot][canonical_global_bone]
LocalBonePool[instance_slot][part_local_bone]
```

Capture writes only the global pool. It does not write the local pool.

Replay immediately gathers one local palette per active instance slot:

```text
slot 0 -> gather PartLocalToGlobal from GlobalBonePool[0] into LocalBonePool[0]
slot 1 -> gather PartLocalToGlobal from GlobalBonePool[1] into LocalBonePool[1]
...
```

Then `redirect_cb1_cs` patches every supported slot in the fake cb1:

```text
cb1[0 * 16 + 5].xy -> LocalBonePool slot 0 current/previous offsets
cb1[1 * 16 + 5].xy -> LocalBonePool slot 1 current/previous offsets
cb1[2 * 16 + 5].xy -> LocalBonePool slot 2 current/previous offsets
```

The replay draw can keep the game's instancing shape when the generated
3Dmigoto command supports it:

```ini
drawindexedinstanced = <index_count>,INSTANCE_COUNT,<first_index>,0,FIRST_INSTANCE
```

`INSTANCE_COUNT` and `FIRST_INSTANCE` are treated only as draw command
placeholders until proven otherwise. Do not generate runtime logic that depends
on `if INSTANCE_COUNT` or other unverified conditional syntax.

This design supports:

```text
same replacement geometry for every same-character instance
different captured pose per SV_InstanceID/cb1 instance slot
```

It does not yet support:

```text
different replacement geometry per instance inside the same instanced draw
```

That harder case is intentionally out of scope for the first multi-instance
runtime.

## Instance Slot Assignment

The observed no-mod two-character dump showed early shadow capture as repeated
single-instance sequences, while later visible draws can use
`DrawIndexedInstanced(..., InstanceCount:2, ...)`.

First supported slot assignment:

```text
frame reset:
  active_capture_slot = -1

first capture IB of a complete character shadow sequence:
  active_capture_slot += 1

each capture IB in that sequence:
  record bones into GlobalBonePool[active_capture_slot]

last capture IB of that sequence:
  mark GlobalBonePool[active_capture_slot] complete/valid
```

The analyzer must identify the first and last capture IBs from the validated
shadow sequence, not from hardcoded shader hashes alone. The runtime must cap
the slot count at generated `MAX_INSTANCE_SLOTS`. Overflow must skip/diagnose
instead of wrapping and overwriting slot 0.

Before enabling generated multi-instance output, FrameAnalysis must verify that
capture sequence order matches the visible `SV_InstanceID`/cb1 slot order. Good
evidence includes native `cb1[slot * 16 + 5].xy`, object/root matrix rows, and
draw order. If the analyzer cannot bind capture slots to replay slots, it must
stop rather than guess.

## Frame Reset And Validity

Frame-boundary reset is required. Runtime state must not carry stale bone data
across frames.

Preferred reset point is frame end, after all capture/replay work for the frame
has completed. Generated INI should call a small reset command list from
`[Present]` or the equivalent stable frame-boundary hook supported by the target
loader:

```ini
[Present]
run = CommandList_BMC_FrameEndReset
```

Frame-start reset is acceptable only when it is guaranteed to run before the
first capture draw of the next frame. The generated runtime must reset:

```text
active_capture_slot
capture sequence progress
slot valid/complete flags
slot frame/generation ids
per-slot captured-record masks, if generated
local-pool valid flags
diagnostic overflow/missing-capture counters
```

Capture may leave matrix rows untouched for performance, but replay must check
validity before using a slot. A slot is drawable only when:

```text
slot_valid == true
slot_frame_id == current_frame_id
required capture records for the replay plan are complete
```

If any required slot or capture record is missing, generated runtime behavior
should skip the replacement draw or fall back to the original draw, depending on
the host branch. It must never silently draw with previous-frame matrices.

Local pools are transient replay products. They should be considered invalid
after each gather/draw phase or after frame reset.

## LOD Capture Under Multi-Instance

LOD does not get a separate global pool.

Main and LOD capture maps have the same runtime shape. Both write into:

```text
GlobalBonePool[instance_slot][canonical_global_bone]
```

The only semantic difference is the map:

```text
main source local bone -> canonical global bone
lod source local bone  -> one or more canonical global bones
```

After the required capture records are complete, replay is identical for main
and LOD because exported parts only read canonical global bones through
`PartLocalToGlobalBoneMap`. A single partial LOD capture record is not enough
unless it covers every canonical global bone required by the replayed part.

LOD and main IB hashes usually differ. Runtime generation must bridge them
explicitly:

```text
LOD TextureOverride key -> canonical exported resource suffixes
LOD capture key         -> LodCaptureBoneMap record
canonical suffix        -> PartLocalToGlobalBoneMap + exported VB/IB buffers
```

The first relationship is geometry ownership, not a bone-palette result. It is
derived before INI generation from cheap LOD/main structure signatures:

```text
used VB2 bone-slot count
BLENDWEIGHTS influence distribution
IB/VB size and AABB metadata
```

When these signatures are close, the LOD IB is considered the corresponding
runtime host for that main exported region. Bone-cloud or point-cloud matching
may only be used for unresolved/ambiguous LOD candidates; it must not broaden a
signature-resolved LOD source to unrelated main regions.

The second relationship is bone scatter:

```text
LOD local bone -> canonical global bone
```

That scatter still uses the LOD's own capture palette and the shared canonical
global pool. It is intentionally independent from the geometry ownership
mapping, because LOD and main meshes can merge or split visual regions
differently.

The LOD override hosts replay with the LOD hash, but the draw resources stay the
main high-detail exported resources. If a canonical main part maps to multiple
LOD sources, only the resolved LOD source for that main region hosts visible
replay; unrelated LOD sources are capture-only. This prevents duplicate
high-detail draws while still allowing all LOD scatter records to fill the
shared global pool.

LOD shadow replay is scheduled separately from visible LOD replay. Visible replay
uses the resolved LOD source for the exported main part. Shadow replay waits
until the final LOD shadow host, and the preceding capture cluster must cover
every canonical global bone required by the delayed replay parts. If coverage
cannot be proven, generated output must keep LOD entries capture-only for the
shadow branch and must not skip the native LOD shadow.

Main and LOD capture maps should remain separate files for clarity:

```text
MainCaptureBoneMap.buf
LodCaptureBoneMap.buf
```

## Multi-Instance Risks And Guards

Known risks:

```text
capture sequence order may not match visible SV_InstanceID order
some capture records may be missing for one slot
MAX_INSTANCE_SLOTS may be exceeded
fake cb1 may not expose enough rows for slot * 16 + 5
current and previous offsets must be patched as a pair
LOD one-to-many mappings can be ambiguous
fixed slot loops add CS cost if not guarded by active slot count
```

Required guards:

```text
only loop active slots when possible
quick-return on invalid slots
validate cb1 row capacity for MAX_INSTANCE_SLOTS
validate capture completion before replay
emit analyzer diagnostics for ambiguous slot binding
keep the old single-instance fast path when MAX_INSTANCE_SLOTS == 1
```

## Multi-Instance FrameAnalysis Questions

When new dumps are available, inspect these fields for the same IB across the
two instances:

```text
DrawIndexed vs DrawIndexedInstanced
InstanceCount
StartInstanceLocation / FIRST_INSTANCE
first index / index count / vertex offsets
cb1 first_constant
shader-visible cb1[slot * 16 + 5].xy
vs-t0 hash and whether cb1[slot * 16 + 5].xy points to distinct windows
world/root/object matrices in nearby CB rows
draw order of capture stage versus replay host
whether each instance has a separate shadow draw sequence
whether capture sequence order matches visible SV_InstanceID order
```

Useful stable instance discriminators, in priority order:

```text
distinct native cb1[slot * 16 + 5].xy bone windows
distinct FirstInstance / SV_InstanceID path
distinct object/root matrix rows
draw order within a complete per-character shadow cluster
```

Generation must stop rather than guess if the analyzer cannot bind every
captured source draw and replay draw to the same instance slot.
