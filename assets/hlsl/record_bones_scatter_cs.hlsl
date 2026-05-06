// =========================================================
// record_bones_scatter_cs.hlsl
// LOD palette capture using scatter pairs.
//
// IniParams[100].x = LOD capture record index
// LodCaptureMeta[record*4 + 0] = pair_base
// LodCaptureMeta[record*4 + 1] = pair_count
// LodCapturePairs[(pair_base + pair)*2 + 0] = lod local bone
// LodCapturePairs[(pair_base + pair)*2 + 1] = canonical global bone
//
// Each pair writes current and previous matrices into the canonical global
// BoneStore pool. Multiple canonical globals may read from the same LOD local.
// =========================================================

StructuredBuffer<uint4> OriginalT0 : register(t0);
StructuredBuffer<uint4> DumpedCB1  : register(t1);
Buffer<uint> LodCaptureMeta        : register(t2);
Buffer<uint> LodCapturePairs       : register(t3);
Texture1D<float4> IniParams        : register(t120);

RWStructuredBuffer<uint4> FakeT0_UAV : register(u1);

static const uint GLOBAL_RESERVED_ROWS = 3;
static const uint PREVIOUS_ROW_OFFSET = 100000;
static const uint CAPTURE_RECORD_INIPARAM = 100;
static const uint LOD_META_STRIDE = 4;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint pair_row = tid.x;
    uint record_index = (uint)IniParams[CAPTURE_RECORD_INIPARAM].x;
    uint meta_base = record_index * LOD_META_STRIDE;
    uint pair_base = LodCaptureMeta[meta_base + 0];
    uint pair_count = LodCaptureMeta[meta_base + 1];
    uint rows_to_copy = pair_count * 3;
    if (pair_row >= rows_to_copy)
    {
        return;
    }

    uint pair_index = pair_row / 3;
    uint row_in_bone = pair_row % 3;
    uint pair_offset = (pair_base + pair_index) * 2;
    uint lod_local_bone = LodCapturePairs[pair_offset + 0];
    uint canonical_global_bone = LodCapturePairs[pair_offset + 1];

    uint src_current_base = DumpedCB1[5].x;
    uint src_previous_base = DumpedCB1[5].y;

    uint src_current_row = src_current_base + GLOBAL_RESERVED_ROWS + lod_local_bone * 3 + row_in_bone;
    uint src_previous_row = src_previous_base + GLOBAL_RESERVED_ROWS + lod_local_bone * 3 + row_in_bone;
    uint dst_current_row = GLOBAL_RESERVED_ROWS + canonical_global_bone * 3 + row_in_bone;
    uint dst_previous_row = PREVIOUS_ROW_OFFSET + dst_current_row;

    FakeT0_UAV[dst_current_row] = OriginalT0[src_current_row];
    FakeT0_UAV[dst_previous_row] = OriginalT0[src_previous_row];
}
