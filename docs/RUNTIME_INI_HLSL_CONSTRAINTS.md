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

`filter_index = 200` means only:

```text
this draw belongs to the early shadow/capture stage
```

It must not encode normal shadow versus transparent shadow. Those are separated
offline by the replay plan and by the IB/collection classification.

Do not generate broad VS override tables for visible, outline, material, or
effect passes outside the verified early shadow window. Visible replacement must
inherit the game's current pass state through `TextureOverride` branches.

## TextureOverride Branch Contract

Each relevant IB override should have only two runtime branches:

```ini
if vs == 200
  ; capture native bones and, only when this source has exported shadow
  ; replacement geometry, skip the original shadow draw.
endif

if vs != 200
  ; visible/outline/material/effect replacement path under current game state.
endif
```

Avoid many pass-specific `if` branches. The INI should be table-driven by
static buffers and by export/replay lists.

`handling = skip` is generated only when the export collection actually contains
replacement geometry for that source/host context. Capturing an IB does not by
itself imply skipping it.

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
if vs == 200
  x100 = <capture_record_index>
  ; bind/select capture map
  run = CustomShader_RecordBones
  ; optional handling = skip
endif

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

After capture, replay is identical for main and LOD because exported parts only
read canonical global bones through `PartLocalToGlobalBoneMap`.

LOD and main IB hashes usually differ. Runtime generation must bridge them
explicitly:

```text
LOD TextureOverride key -> canonical exported resource suffixes
LOD capture key         -> LodCaptureBoneMap record
canonical suffix        -> PartLocalToGlobalBoneMap + exported VB/IB buffers
```

The LOD override hosts replay with the LOD hash, but the draw resources stay the
main high-detail exported resources. If a canonical main part maps to multiple
LOD sources, only the strongest resolved LOD source hosts replay; the remaining
LOD sources are capture-only. This prevents duplicate high-detail draws while
still allowing all LOD scatter records to fill the shared global pool.

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
