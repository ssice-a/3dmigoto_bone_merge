# BMC LOD Recognition Constraints

This document records the LOD recognition rules only. Runtime INI emission,
shadow replay layout, and export draw ordering are intentionally left to the
runtime/export constraints document.

## Core Rule

An IB hash is only a draw-match entry point. It is not a stable identity for a
bone layout, a palette, or a main/LOD relationship.

LOD recognition must build explicit relationships between:

- LOD draw chains
- LOD IB keys
- main IB keys
- LOD capture records
- canonical global bone slots

These relationships must be kept separate. A correct LOD-to-main hash link does
not prove that the LOD capture palette covers every global bone needed by replay.

## Runtime Stages

BMC runtime behavior has three conceptual stages. LOD and main/control draws use
the same stages; only the capture profile and chain data differ.

### 1. Record Stage

The record stage captures bone matrices from the currently bound game resources
and writes them into the canonical `GlobalBonePool`.

This stage is **profile-sensitive**.

The active capture profile decides which capture map interprets the current
draw's local bone indices:

```ini
run = CustomShader_ExtractCB1
x100 = ...
cs-t2 = ResourceMainCaptureBoneMap ; or ResourceLodCaptureBoneMap
run = CustomShader_RecordBones
```

`x100` is a record index inside the bound capture map. It is not a global bone
base. The actual writes are defined by the map's pairs:

```text
source_local_bone -> canonical_global_bone
```

The same override key may need different record logic when it can execute in
different profiles. In that case the capture map must be selected by a profile
flag or an equivalent profile discriminator.

### 2. Delayed Shadow Replay Stage

The delayed shadow replay stage runs at the final host draw of a recognized
shadow/capture chain. Its purpose is to wait until the chain has captured enough
bone data, then replay exported parts in the shadow pass.

This stage is **chain/host-sensitive**.

It decides:

- which source shadow draws are skipped
- which host receives delayed replay
- whether the host's original draw must be preserved with `draw = from_caller`
- which exported parts are replayed on the host

It does not decide how local bone indices are interpreted. That is record-stage
work.

### 3. Main Visible Replay Stage

The visible replay stage runs during normal visible rendering. It skips the
source draw, gathers the exported part's local palette from `GlobalBonePool`,
redirects cb1 to the local pool, binds exported IB/VB resources, and draws.

This stage is **part/replay-sensitive**.

```ini
cs-t2 = ResourcePartLocalToGlobalBoneMap_...
run = CustomShader_GatherLocalBones
vs-t0 = ResourceLocalBonePool_SRV
run = CustomShader_RedirectCB1
vs-cb1 = ResourceFakeCB1
drawindexedinstanced = ...
```

The visible replay stage uses the exported part's
`PartLocalToGlobalBoneMap`. It does not branch on main/LOD profile. Both main
and LOD replay consume the same canonical global bone pool.

## Profile Discrimination

Profile discrimination belongs to the record stage.

If an override key can only appear in one capture profile, it can use a direct
record block:

```ini
x100 = MAIN_RECORD
cs-t2 = ResourceMainCaptureBoneMap
run = CustomShader_RecordBones
```

or:

```ini
x100 = LOD_RECORD
cs-t2 = ResourceLodCaptureBoneMap
run = CustomShader_RecordBones
```

If an override key or host section can execute in multiple capture profiles,
recording must branch on an explicit context flag, such as:

```ini
[Constants]
global $bmc_profile_lod = 0

; At the first draw of a recognized LOD capture chain:
if vs == 200
  $bmc_profile_lod = 1
endif

if $bmc_profile_lod == 1
  run = CustomShader_ExtractCB1
  x100 = LOD_RECORD
  cs-t2 = ResourceLodCaptureBoneMap
  run = CustomShader_RecordBones
else
  run = CustomShader_ExtractCB1
  x100 = MAIN_RECORD
  cs-t2 = ResourceMainCaptureBoneMap
  run = CustomShader_RecordBones
endif

; At the final host of that same LOD capture chain:
if vs == 200
  $bmc_profile_lod = 0
endif
```

The flag chooses how to write bones into `GlobalBonePool`. It must not be used
as the authority for which exported parts to replay.

Emit this flag only when an override key can be reached by both main and LOD
record profiles. Keys that exist in only one profile keep direct capture blocks.

## Terms

- **Override key**: `(ib_hash, match_first_index, match_index_count)`.
- **Main key**: an override key from the main/control capture.
- **LOD key**: an override key from the LOD frame analysis.
- **LOD chain**: a contiguous LOD pass group in the frame analysis, usually a
  consecutive `vs == 200` shadow/capture sequence for the same character.
- **LOD host**: the last effective draw in an LOD chain where delayed replay
  should be attached.
- **Capture profile**: the concrete bone layout currently bound by a draw. The
  same IB hash may have different capture profiles in main and LOD contexts.

## Recognition Pipeline

1. Parse the LOD frame analysis into draw records.
2. Identify LOD `vs == 200` chains before doing main/LOD matching.
3. For each chain, collect all candidate LOD keys in draw order.
4. Select the chain host from the chain itself, not from a global singleton.
5. Match main keys only against LOD keys that belong to the relevant chain.
6. Build LOD capture records independently from the main/LOD hash links.
7. Validate that the chain can write every canonical global bone required by
   exported replay parts.

Runtime export may produce more than one LOD shadow replay plan. Each plan is
keyed by the final host of its own recognized capture chain. A global
`lod_manifest_snapshot.shadow_stage.host_*` value is diagnostic context only; it
must not be treated as the sole replay host when the frame contains several LOD
capture chains or a later composite draw.

The analyzer must carry recognized chains forward as explicit data. At minimum
a chain records:

```text
chain_index
draw_start / draw_end
start_lod_record_key / start_key
host_lod_record_key / host_key
lod_record_keys
```

Runtime generation uses this data for profile markers and delayed shadow host
selection.

## Chain Detection

LOD chain detection must be based on draw order and pass state. A chain should
only include draws that are part of the same LOD capture/shadow pass.

Useful signals include:

- `vs == 200` after shader override filtering
- the bone-store VS hash family
- compatible pixel shader/pass role
- consecutive draw order before the render pipeline changes
- matching character/resource context when available

Do not select a fixed global LOD shadow host such as `82254888` or `ef95f8f2`.
Different frames and render paths can end on different hosts. For example, one
frame may end on `ef95f8f2`, while another valid LOD chain may end on
`df4b620c`.

## LOD-to-Main Matching

LOD-to-main matching answers one question only:

> Which exported main part should be replayed when this LOD key is encountered?

The matcher should be chain-scoped:

- A main key may only match LOD keys present in the same recognized LOD chain.
- Matching all main records against all LOD records globally is too broad.
- `vb2` slot count can be a fast first pass, but it is not authoritative.
- `vb2` slot-signature matching is one-to-one. One LOD key must not be claimed
  by several unrelated main keys just because their slot counts are close.
- When there are several recognized chains, matching is executed separately per
  chain. A main key may therefore produce several LOD links, but each emitted
  link contains one concrete LOD source and its chain metadata.
- Geometry center/bounds/diag are supporting evidence.
- Vertex-group point-cloud matching is the source of truth for one-to-many
  relationships. Small vertex-group clouds that fail direct matching may be
  merged into a larger compatible group when the spatial evidence supports that
  relationship.

If two LOD candidates are both plausible, prefer the one in the same chain and
with stronger geometry/slot affinity. Do not let a LOD key be claimed by
multiple unrelated main keys without an explicit multi-source reason.

## Palette And Capture Separation

LOD-to-main matching is independent from bone capture.

The replay part uses its exported `PartLocalToGlobalBoneMap` and therefore reads
from canonical global bone slots. The LOD capture record must write the correct
native LOD local bones into those canonical global slots.

This means:

- LOD links decide replay ownership.
- LOD capture records decide bone data placement.
- The same LOD key can be the correct replay host but still fail if its capture
  record does not cover the required global bones.

## Capture Provider Selection

Capture provider selection must start from exported geometry, not from a fixed
main/LOD IB count.

For every exported part, build its actual bone demand from the vertex weights of
the objects inside that part:

```text
part -> required canonical_global_bones
```

If a collection contains objects from another body region, that is not a special
case. The part still owns those exported vertices, so its required global bones
must include every weighted vertex group used by those objects. Collection names
and object names must not be used to trim the part palette.

For each capture profile, gather the union of all globals needed by the replay
parts that can be drawn in that profile:

```text
MainNeededGlobals = union(main replay part required globals)
LodNeededGlobals  = union(LOD replay part required globals)
```

Then find provider draws independently for each profile:

```text
MainProviders = main capture records that write MainNeededGlobals
LodProviders  = LOD capture records that write LodNeededGlobals
```

A draw participates in capture only if its bound capture map can write at least
one currently needed canonical global bone. If it only writes unused globals, it
should not be required by the replay plan.

This applies to visible-stage refresh as well as shadow-stage capture. Visible
record is not a blanket operation over every BMC override. The exporter must
first know which canonical global bones are used by exported geometry, then
record only the matching provider overrides that can write those globals.

Main and LOD provider counts are allowed to differ. For example, a main replay
set may need four provider IBs, while the corresponding LOD replay set may need
three or five. The counts must not be inherited between profiles; both profiles
share the canonical global bone pool, but their source-local bone layouts and
provider draws are selected separately.

When multiple export collections contain objects, compute each part palette
first, then union those palettes only for provider selection. Replay still uses
the individual part's `PartLocalToGlobalBoneMap`.

## Same Hash, Different Layout

If a hash appears in both main and LOD contexts, or appears in multiple LOD
profiles, do not assume it uses the same bone layout.

The selector must distinguish the capture profile by context, such as:

- containing LOD chain
- pass role
- VS/filter index
- bound `vb0/vb1/vb2` resource signature
- `vs-t0`/CB bone-window signature when available

The generated runtime plan must be able to express separate main and LOD capture
records for the same IB hash when their layouts differ.

For same-key profile conflicts, the profile flag is a chain context marker, not
a new draw identity. The start key of the recognized LOD chain sets it, the
final host key resets it, and frame end resets it defensively.

## Coverage Validation

Before runtime INI generation, every LOD chain that will replay exported parts
must be checked:

1. Collect all `canonical_global_bone` values required by the replay parts.
2. Collect all canonical global bones written by capture records in that chain
   before the selected host.
3. Report missing bones.
4. If missing bones are used by exported geometry, the plan is unsafe.

Unsafe plans must not be silently emitted as if they are correct. By default,
export should be blocked and the diagnostic output must report the missing
globals, affected profile, chain, host, and replay part. An advanced
"allow incomplete export" escape hatch may exist later, but it must not be the
default path.

## Expected Debug Output

LOD analysis should print enough information to diagnose wrong links quickly:

```text
[BMC LOD] chain=0 draws=32-37 host=df4b620c:0:23364 keys=6
[BMC LOD] link df4b620c:0:23364 -> 614a8c60:0:45003 method=slot_signature score=...
[BMC LOD] coverage chain=0 required=230 captured=230 missing=0
```

If coverage is incomplete:

```text
[BMC LOD] coverage chain=0 required=230 captured=219 missing=11
[BMC LOD] missing globals: 85,86,92,155,...
```

## Non-Goals

This document does not define:

- final INI replay block placement
- `draw = from_caller` rules
- main shadow replay rules
- part splitting rules
- runtime HLSL implementation details

Those rules belong in the export/runtime constraint document.
