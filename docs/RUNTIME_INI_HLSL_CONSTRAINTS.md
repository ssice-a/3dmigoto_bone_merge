# Bone Merge Runtime INI/HLSL Constraints

This document is the source of truth for the generated schema v3 runtime.

## Scope

The runtime has two responsibilities:

1. Capture native current/previous bone matrices into one canonical global pool.
2. Gather an exported part palette into a local `vs-t0`, patch the active bone
   window in the native constant buffer, and replay replacement geometry.

Mesh analysis, palette construction, LOD matching, texture selection, and CPU
pre-skinned detection are offline Blender responsibilities.

## Export Collection Contract

The export collection tree owns geometry. Object names are metadata only.

```text
Export Root
  <ib_hash>-<match_index_count>-<first_index>
    direct mesh objects...           -> implicit part00

Export Root
  <ib_hash>-<match_index_count>-<first_index>
    part00
      mesh objects...                -> explicit part00
    part01
      mesh objects...                -> explicit part01
```

Direct meshes share one buffer set and one palette, but retain independent draw
ranges. Once a region contains a `partNN` child, direct meshes on the region are
invalid. Export preparation must not move or relink source objects.

A draw classified as `cpu_pre_skinned` may be imported as reference geometry.
Its object name includes `[CPU_SKINNED_UNSUPPORTED]`, and replacement export
must reject it because no shader-side local bone palette can reproduce it.

## Shader Classification

Generated mods do not emit character VS hash lists. EFMI Core owns the shared
`CharacterShaderClassifier.ini` ShaderRegex rules:

```text
200 = compatible shadow/capture VS, bone-window CB is b1
201 = compatible visible family, bone-window CB is b2
202 = compatible visible family, bone-window CB is b2
203 = compatible visible/effect family, bone-window CB is b2
```

Replacement replay is a positive allowlist:

```ini
if vs == 201 || vs == 202 || vs == 203
  ; replacement replay
endif
```

Unknown and special-effect shaders remain on the native draw path. Do not
restore `if vs != 200`, residual exclusions, per-mod static shader hashes, or a
`filter_index = 204` maintenance list.

## Runtime Architecture

`materialize_bonestore_runtime()` emits:

```text
schema_version       = 3
runtime_architecture = efmi_hashregion_fifo_cb_pool_v1
instance_pool_size   = 8
```

The generated INI contains one FIFO pool:

```ini
[PoolBMCInstanceRegistry]
pool_size = 8
pool_index_type = fifo
pool_lazy_init = false
type = Buffer
format = R32G32B32A32_UINT
array = 4096
```

The pool has two roles:

1. `#PoolBMCInstanceRegistry[$uid]` assigns a stable capture slot for a UID.
2. `PoolBMCInstanceRegistry[$uid] = copy vs-cb1` stores that instance's
   shader-visible shadow CB for delayed replay.

Each physical slot also has INI metadata:

```text
$bmc_slot_uid_N
$bmc_slot_native_N
```

Visible replay resolves UIDs only against these recorded values. It must not use
`#Pool[...]` for lookup, because looking up an unknown UID allocates a FIFO slot
and may evict a valid captured instance.

## FirstConstant Contract

EFMI resource operations already apply the bound constant-buffer view:

- `copy vs-cb1` and `copy vs-cb2` copy the visible region beginning at
  `FirstConstant` into row zero of the destination.
- `vs-cbN->HashRegion(offset, size)` adds `FirstConstant * 16` to `offset`
  before hashing the backing buffer.

The plugin therefore uses only shader-visible offsets:

```ini
$bmc_hash_offset = $bmc_native_slot * 256
$bmc_instance_uid = vs-cb1->HashRegion($bmc_hash_offset, 64)
```

`256` is `16 CB rows * 16 bytes`; `64` hashes the four root/object matrix rows.
Do not add `FirstConstant` in INI or HLSL. After copying, HLSL row
`slot * 16 + 5` is already the correct shader-visible bone-window row.

## Capture Contract

VS200 capture uses `b1` and `vs-t0`:

```ini
if vs == 200
  $uid = vs-cb1->HashRegion(...)
  $capture_slot = #PoolBMCInstanceRegistry[$uid]
  PoolBMCInstanceRegistry[$uid] = copy vs-cb1
  ResourceCapturedCB = ref PoolBMCInstanceRegistry[$uid]
  cs-t2 = ResourceCaptureBoneMap_<source>
  run = CustomShader_RecordBones
endif
```

Each capture map is a `Buffer<R32_UINT>`:

```text
uint[0] = pair_count
uint[1] = source local bone count metadata
uint[2] = target/global count metadata
uint[3] = flags
uint[4 + pair * 2 + 0] = source_local_bone
uint[4 + pair * 2 + 1] = canonical_global_bone
```

For each pair, `record_bones_cs.hlsl` copies three current rows and three
previous rows from native `vs-t0`. The source bases are read from copied CB row
`native_slot * 16 + 5`. Bounds and nonzero data are validated before the slot
is marked valid.

Multiple character parts with the same root-matrix UID write different global
bones into the same capture slot. This is how one character-wide pool is built
without relying on draw order for visible instance assignment.

## Part Palette Contract

Every exported part has one local-to-global map:

```text
uint[0]   = part local bone count
uint[1..] = part local bone -> canonical global bone
```

The local count lives in the buffer. Do not restore `x101` or another INI-side
count. One part may contain multiple Blender objects and draw ranges; they all
share this palette.

## HLSL Modules

The bundled runtime files are:

```text
bone_store_common.hlsli
record_bones_cs.hlsl
clear_instance_mapping_cs.hlsl
resolve_instance_mapping_cs.hlsl
gather_local_bones_cs.hlsl
redirect_cb_cs.hlsl
reset_runtime_state_cs.hlsl
```

There is no point-draw CB extractor. `extract_cb1_vs.hlsl`,
`extract_cb1_ps.hlsl`, `redirect_cb1_cs.hlsl`, and `draw = 4096` belong to the
removed schema v2 path.

`redirect_cb_cs.hlsl` copies the shader-visible CB and patches only:

```text
cb[native_slot * 16 + 5].x = local current base
cb[native_slot * 16 + 5].y = local previous base
```

All other CB data remains native to the current pass or saved shadow instance.

## Shadow Replay

Shadow source draws capture into the global pool. Exported source shadows are
delayed to the final compatible host so all parts have contributed bones.

The FIFO pool stores each captured instance CB because real frames may use:

```text
shadow: instance A in one single-instance draw
shadow: instance B in another single-instance draw
visible: A and B packed into one two-instance draw, possibly in reverse order
```

At the final host, replay iterates occupied metadata slots. For each slot it:

1. Binds `PoolBMCInstanceRegistry[$slot_uid]` as `ResourceCapturedCB`.
2. Resolves only the saved native slot to the physical capture slot.
3. Gathers each exported part from that global slot.
4. Patches `b1`, then draws one instance using the saved native instance index.

Transparent shadow batches run before the white shadow texture is bound. Normal
shadow batches run afterward. A non-replaced final host is preserved once with
`draw = from_caller`.

The per-draw capacity guard is:

```ini
if first_instance + instance_count <= 8
  ; skip/delay/replay path
endif
```

A draw outside this range keeps its native rendering. The fixed pool should be
raised only together with INI loops and all HLSL constants.

## Visible Replay

VS201-203 replay copies the current `b2` visible region, hashes each native
instance root matrix, and resolves it against captured slot metadata. It then
gathers all mapped slots and keeps the game's instanced draw shape:

```ini
ResourceCapturedCB = copy vs-cb2
run = CustomShader_ClearInstanceMapping
; resolve current UIDs against $bmc_slot_uid_N

if $bmc_mapping_valid == 1
  handling = skip
  ; gather part, patch b2, bind exported buffers
  drawindexedinstanced = INDEX_COUNT,INSTANCE_COUNT,FIRST_INDEX,0,FIRST_INSTANCE
endif
```

If any active instance lacks a captured UID, `$bmc_mapping_valid` becomes zero
and `handling = skip` is not executed. The native draw is the safety fallback.

## Geometry Binding

Every recorded vertex-buffer slot is exported and rebound in numeric slot
order. If a layout has no independent `vb3`, runtime aliases `vb3` to `vb0`.
An independent `vb3` is preserved as its own resource. CPU pre-skinned draws
are rejected before replacement export.

The usual game layout is:

```text
vb0 = position
vb1 = texcoord/normal/tangent data
vb2 = blend data
vb3 = vb0 alias when no independent stream exists
ib  = R32_UINT exported index data
```

## LOD Contract

LOD recognition remains offline and chain-scoped. Main and LOD capture maps may
use different source-local palettes, but both scatter into the same canonical
global pool and use the same UID/CB-pool runtime.

Every recognized LOD chain owns its actual final shadow host. Shadow replay is
enabled only after coverage validation proves that the chain can populate all
canonical bones used by its exported parts. Automatic LOD behavior is not a
separate runtime path.

## Frame Reset

`[Present]` performs one reset command list:

```ini
PoolBMCInstanceRegistry = null
run = CustomShader_ResetRuntimeState
$bmc_slot_uid_N = -1
$bmc_slot_native_N = -1
```

Matrix buffers are not cleared every frame. Runtime validity flags prevent stale
rows from being consumed, which avoids unnecessary large GPU clears.

## Known Limits

The stable UID is the four-row root/object matrix. This correctly maps the
observed same-character double-instance frame even when shadow order and visible
`SV_InstanceID` order differ.

Two simultaneous instances with byte-identical root matrices produce the same
UID. If they also have different poses, the later capture can overwrite the
same slot. The current INI language cannot form a character-wide identity from
bone data while keeping multiple parts in one canonical slot, so this remains a
documented protected limit rather than a guessed mapping.

Other limits:

- At most eight native instance indices (`0..7`) participate in replacement.
- One generated mod uses one independent pool and one frame reset hook.
- Different replacement geometry per instance inside one instanced draw is not
  supported.
- ShaderRegex must continue excluding shaders with incompatible CB layouts.
