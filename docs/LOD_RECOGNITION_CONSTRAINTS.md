# BMC LOD Recognition Constraints

This document records offline LOD recognition rules. Runtime resource and
replay semantics are defined in `RUNTIME_INI_HLSL_CONSTRAINTS.md`.

## Core Rule

An IB hash is a draw-match entry point, not a stable bone-layout identity.
Recognition must keep these relationships explicit and separate:

- LOD draw chain
- LOD override key
- main override key
- LOD capture map
- canonical global bone slots
- final chain shadow host

A correct LOD-to-main geometry link does not prove capture coverage.

## Runtime Stages

Main and LOD use the same schema v3 stages.

### Record

VS200 capture hashes the shader-visible root matrix in `b1`, stores the visible
CB in the FIFO instance pool, and scatters native `vs-t0` rows through the
selected capture map:

```ini
$uid = vs-cb1->HashRegion($native_slot * 256, 64)
$capture_slot = #PoolBMCInstanceRegistry[$uid]
PoolBMCInstanceRegistry[$uid] = copy vs-cb1
cs-t2 = ResourceCaptureBoneMap_<main_or_lod_source>
run = CustomShader_RecordBones
```

The concrete capture map, rather than an INI record index, defines:

```text
source_local_bone -> canonical_global_bone
```

If one override key is present in both main and LOD contexts, all required
capture maps may run for that key. They scatter into the same canonical pool;
runtime does not guess a profile from IB hash alone.

### Delayed Shadow Replay

Each recognized chain owns its final shadow host. Source shadows for exported
parts are delayed until that host so the canonical pool has complete coverage.
The FIFO CB pool replays each captured UID independently.

The host decision controls:

- which source shadows are skipped
- whether the host needs `draw = from_caller`
- transparent versus normal replay order
- which exported main geometry represents the LOD key

### Visible Replay

VS201-203 use `b2`, resolve the current root-matrix UID against captured slot
metadata, gather the exported part palette, and replay canonical exported
geometry. LOD does not have a second visible HLSL path.

If a visible instance has no captured UID, the native draw is preserved.

## Terms

- **Override key**: `(ib_hash, match_first_index, match_index_count)`.
- **Main key**: an override key from the main/control frame.
- **LOD key**: an override key from the LOD frame.
- **LOD chain**: a contiguous compatible VS200 capture sequence.
- **LOD host**: the last effective draw in one LOD chain.
- **Capture provider**: a draw whose palette can populate canonical bones.
- **Replay link**: an offline LOD key -> exported main key relationship.

## Recognition Pipeline

1. Parse LOD draws and their exact IB region keys.
2. Partition VS200 draws by root-matrix instance signature.
3. Group those instances by character identity and select the target group.
4. Build compatible chains and one host per selected target instance.
5. Match selected LOD keys to main keys with layout and bone evidence.
6. Build capture pairs independently from replay links.
7. Determine which canonical bones exported replay actually requires.
8. Add same-chain capture providers for missing bones.
9. Enable shadow skipping only after identity and coverage validation.

Runtime export may emit multiple LOD shadow replay plans. A global singleton
host is invalid when a frame contains multiple character chains.

## Blender Profile Ownership

Blender stores each analyzed directory as one LOD Profile with its own level,
FrameAnalysis path, chains, links, capture records, mapping, review, and
snapshot. Enabled profiles are aggregated only when their recorded global-pool
generation matches the current pool.

Changing a profile level or directory, or rebuilding a different global pool,
marks that analysis stale. Stale enabled profiles block export until they are
reanalyzed. Disabling a profile removes it from the generated aggregate without
deleting its stored analysis.

Draw indices are local to a FrameAnalysis directory. Chain detection, automatic
donor completion, and coverage review must therefore stay scoped by
`lod_profile_id`; equal draw indices in two profiles are unrelated.

The Mapping list is diagnostic. Fallback suggestions are applied only when the
corresponding `(lod_profile_id, canonical_global_bone)` row remains enabled in
the Blender repair list. Unselected or unresolved used bones continue to block
export.

If two enabled profiles use the same override key and map one canonical target
from different source-local bones, export must stop. INI runtime identity cannot
distinguish those profile states safely.

## Character Identity Selection

A root-matrix signature identifies one runtime instance in the captured frame;
it does not identify which character model owns that instance. Different
humanoid characters can have nearly identical bone counts, slot signatures,
and point clouds. Those similarities are valid bone-mapping evidence only
after the target character identity has been selected.

The analyzer groups chains by root-matrix signature, then looks for an exact
main override key inside each group. The strongest exact main IB region is the
automatic identity anchor. Every instance group containing that same anchor is
selected, which keeps same-character multi-instance captures while excluding
other characters in the scene.

When a frame has one root-matrix signature, that sole group is unambiguous. If
a frame has multiple signatures and none contains an exact main identity
anchor, analysis must stop. Bone-cloud similarity must never choose a character
identity. Capture a frame with one unchanged target part or isolate the target
instead of exporting a guess.

The result records selected and excluded signatures in `lod_identity`. Only
selected chains are copied into `lod_chains`, `lod_links`, capture records, and
runtime shadow plans. Analyzer results from before this identity partition are
stale.

## Chain Detection

Chain detection first hashes the four shader-visible root-matrix rows from each
VS200 `b1` view. The byte offset is:

```text
(FirstConstant + (FirstInstance + instance_offset) * 16) * 16
```

Draws with different root-matrix signatures belong to different logical
instances even when they are adjacent and share the same backing CB or
`vs-t0`. Ordering and local continuity are applied only after instance
partitioning and target character selection.

Useful boundaries include:

- transition out of the VS200 capture stage
- large draw-index gaps
- incompatible CB or `vs-t0` contracts
- a new recognized character cluster
- a completed final host followed by another capture sequence

The configured chain-gap threshold is a conservative fallback, not identity.
Stored LOD results from analyzers that predate root-matrix identity are stale
and must be analyzed again before export.

## LOD-To-Main Matching

Preferred evidence, strongest first:

1. Exact compatible vertex-layout and weighted bone signature.
2. Point-cloud/bone-cloud correspondence from decoded geometry.
3. Stable per-slot blend signatures.
4. Same-part donors already established in the chain.

IB hash equality alone is insufficient. Dynamic VB0 identity is also
insufficient because the same allocation may be reused.

One LOD local bone may scatter to multiple canonical globals when evidence
proves a one-to-many relationship. Conversely, conflicting candidates must not
be collapsed only to satisfy coverage.

## Palette And Capture Separation

Replay geometry always uses the exported main part's
`PartLocalToGlobalBoneMap`. LOD capture maps only decide where native LOD rows
land in the canonical pool.

```text
LOD source local palette -> canonical global pool -> exported main local palette
```

Do not bind a LOD capture map as an exported part palette, and do not use an
exported part map to interpret native LOD source-local indices.

## Capture Provider Selection

A replay-linked LOD key may not expose every required bone. Missing bones can
be inherited only from compatible providers in the same recognized chain.

Provider selection must record:

- provider override key
- source local bone
- target canonical global bone
- evidence type
- chain ownership

Sparse/noisy candidates that broaden relationships without geometry or blend
support must be rejected.

## Same Hash, Different Layout

The same IB hash may appear with different first-index regions, layouts, or
capture palettes. All lookup tables must use the full override key. A fallback
from `(hash, first_index, count)` to hash-only matching is unsafe.

## Coverage Validation

Before enabling delayed LOD shadow replacement, prove:

```text
required canonical globals for replay
    subset of
canonical globals written by all accepted capture providers in this chain
```

The provider union is scoped to the chain's root-matrix signature. Coverage
from another instance cannot make the current chain appear complete.

If proof fails:

- keep native shadow draws
- report missing canonical globals and candidate providers
- allow explicit fallback repair in Blender
- do not emit a partial replacement shadow

At runtime an LOD host hashes its current `b1` root matrix and replays only the
capture slot with the same UID. It never enumerates every occupied global slot.
If two instance chains share one static host key but require different exported
part sets, delayed replacement for that host is disabled and native shadows are
kept.

Unused unmatched groups do not block export. Only bones actually referenced by
exported weighted vertices contribute to required coverage.

## Expected Debug Output

Analysis diagnostics should include:

- chain id and draw range
- selected host key and draw index
- each LOD -> main replay link with evidence
- every capture pair and provider
- required, covered, and missing canonical bones
- rejected ambiguous/noisy candidates
- whether shadow replay is enabled or kept native

## Non-Goals

LOD recognition does not:

- classify character shaders; EFMI Core ShaderRegex owns that
- distinguish byte-identical root-matrix instances
- replace CPU pre-skinned draws
- choose different replacement geometry per instance
- infer missing capture coverage at runtime
