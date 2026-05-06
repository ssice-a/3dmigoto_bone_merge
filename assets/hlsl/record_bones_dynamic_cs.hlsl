// =========================================================
// record_bones_dynamic_cs.hlsl
// Main palette capture using a compact static table.
//
// IniParams[100].x = capture record index
// CaptureMeta[record*4 + 0] = capture_store_base
// CaptureMeta[record*4 + 1] = compact_bone_count
// CaptureMeta[record*4 + 2] = source_local_index_base
// CaptureLocalIndices[source_local_index_base + compactBone] = native local bone
//
// Source palette layout:
//   cb1[5].x/y + 3 + nativeLocalBone*3 + {0,1,2}
// Destination palette layout:
//   3 + (capture_store_base + compactBone)*3 + {0,1,2}
// =========================================================

StructuredBuffer<uint4> OriginalT0 : register(t0);
StructuredBuffer<uint4> DumpedCB1  : register(t1);
Buffer<uint> CaptureMeta          : register(t2);
Buffer<uint> CaptureLocalIndices  : register(t3);
Texture1D<float4> IniParams       : register(t120);

RWStructuredBuffer<uint4> FakeT0_UAV : register(u1);

static const uint GLOBAL_RESERVED_ROWS = 3;
static const uint PREVIOUS_ROW_OFFSET = 100000;
static const uint CAPTURE_RECORD_INIPARAM = 100;
static const uint CAPTURE_META_STRIDE = 4;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint local_row = tid.x;
    uint record_index = (uint)IniParams[CAPTURE_RECORD_INIPARAM].x;
    uint meta_base = record_index * CAPTURE_META_STRIDE;
    uint capture_store_base = CaptureMeta[meta_base + 0];
    uint bone_count = CaptureMeta[meta_base + 1];
    uint local_index_base = CaptureMeta[meta_base + 2];
    uint rows_to_copy = bone_count * 3;
    if (local_row >= rows_to_copy)
    {
        return;
    }

    uint compact_bone = local_row / 3;
    uint row_in_bone = local_row % 3;
    uint native_local_bone = CaptureLocalIndices[local_index_base + compact_bone];

    uint src_current_base = DumpedCB1[5].x;
    uint src_previous_base = DumpedCB1[5].y;

    uint src_current_row = src_current_base + GLOBAL_RESERVED_ROWS + native_local_bone * 3 + row_in_bone;
    uint src_previous_row = src_previous_base + GLOBAL_RESERVED_ROWS + native_local_bone * 3 + row_in_bone;
    uint dst_current_row = GLOBAL_RESERVED_ROWS + (capture_store_base + compact_bone) * 3 + row_in_bone;
    uint dst_previous_row = PREVIOUS_ROW_OFFSET + dst_current_row;

    FakeT0_UAV[dst_current_row] = OriginalT0[src_current_row];
    FakeT0_UAV[dst_previous_row] = OriginalT0[src_previous_row];
}
