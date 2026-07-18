# Vertex Layout Round-Trip Contract

## Goal

Importing a captured mesh and exporting it without edits must preserve every
non-skin vertex field byte-for-byte. Skin fields may change only when the
exported Part palette intentionally remaps source bone indices; their weighted
bone semantics must remain equal. These invariants apply to every candidate
layout, not to a list of known character hashes.

Blender expands exported geometry to loop vertices, so the output buffer can
have a different record count and order. Exactness is defined per output loop:
its bytes must equal the source record of that loop's imported point, except
for a proven Part-palette remap.

## Layout Model

The core.vertex_layout_codec module owns the vertex layout vocabulary:

- A semantic field is one input-assembler interpretation such as TEXCOORD4.
- A physical field is one byte range in one vertex stream.
- Several semantic fields can alias one physical field.
- A source identity records the captured resource, backing buffer, offset,
  stride, and vertex count.

Exact aliases such as TEXCOORD2 and TEXCOORD4 are written once. Partially
overlapping fields are rejected until the codec has an explicit policy for
that layout. Vertex streams are aliased only when their captured source
identities match; equal-looking layouts are not sufficient evidence.

## Blender Carriers

Import creates two representations:

1. Editable semantic carriers for positions, normals/tangent frames, UVs,
   weights, colors, and supported auxiliary values.
2. Lossless point-domain bmc_raw_vb*_u32_* carriers containing each complete
   source record, including padding and unsupported fields.

Four-byte values stored in BYTE_COLOR use Blender's color_srgb channel. Using
color would apply color-space conversion and cannot represent all 256 byte
values exactly.

The Blender audit also checks editable carriers directly, before export. It
compares normal, tangent, handedness, UV, byte-color, and auxiliary numeric
carriers independently from the raw bytes. Blender loop normals are reported
separately as a viewport advisory: Blender's custom-normal space can return a
zero loop normal for a source direction it cannot represent. The explicit
normal carrier remains authoritative and its source bytes are still covered by
the raw-carrier test.

## Export Policy

Every output record begins as a copy of its lossless raw carrier. The exporter
then considers each physical field once:

- If an editable value changed semantically, the field is re-encoded.
- If Blender only introduced an equivalent normalization or quantization
  change, the original bytes remain.
- Unknown fields and padding remain byte-identical.
- If a field cannot be reconstructed and no raw carrier exists, export stops
  with a field-specific error.

There are no neutral zero defaults for required captured fields. A default is
allowed only when a shader contract explicitly defines it.

For packed NORMAL0, semantic equivalence compares the decoded normal, tangent,
and handedness rather than packed integers. Octahedral coordinates have
encoding seams where very different integers represent the same vector. For
skin data, equivalence compares the aggregate weight per bone, so channel
reordering and duplicate source indices do not create false edits.

The audit resolves every exported blend index through the generated Part
palette before comparing aggregate weight per source group. A byte difference
without that semantic proof remains a failure. This exception applies only to
physical fields whose aliases are all BLENDWEIGHTS or BLENDINDICES; padding and
every non-skin field in the same vertex stream must remain byte-identical.

## Format Evolution

DXGI format size, dtype, component count, and normalization live in one
registry. A newly observed format follows one of three paths:

1. Add an editable adapter and its quantization rules.
2. Preserve it through the raw carrier while reporting it as non-editable.
3. Reject the layout if even its physical size or overlap is ambiguous.

Adding character-specific IB or VS hashes is not a format adapter.

## Verification

tests/blender_vertex_roundtrip.py loads every candidate in a capture through
Blender, exports it, and reports byte and field mismatches. CPU pre-skinned
reference draws are reported separately because replacement export is
intentionally unsupported.

The Python unit suite covers format registration, physical aliases, stream
identity, raw preservation, missing-carrier failures, color channels, and
semantic edit paths. The Blender audit is the acceptance test for a real
capture.
