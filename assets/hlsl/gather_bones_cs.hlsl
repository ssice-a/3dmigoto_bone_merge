// =========================================================
// gather_bones_cs.hlsl
// Local palette gather:
//   t0 = Shared native capture store / BoneStore SRV
//   t2 = LocalPalette buffer, palette[localBone] = captureStoreIndex
//   IniParams[101].x = local_bone_count
//
// Capture store layout:
//   current:  3 + captureStoreIndex*3 + {0,1,2}
//   previous: 100000 + 3 + captureStoreIndex*3 + {0,1,2}
//
// Local gathered palette layout:
//   current:  3 + localBone*3 + {0,1,2}
//   previous: 1024 + 3 + localBone*3 + {0,1,2}
// =========================================================

StructuredBuffer<uint4> GlobalFakeT0 : register(t0);
Buffer<uint> LocalPalette            : register(t2);
Texture1D<float4> IniParams          : register(t120);

RWStructuredBuffer<uint4> LocalFakeT0_UAV : register(u1);

static const uint GLOBAL_RESERVED_ROWS = 3;
static const uint GLOBAL_PREVIOUS_ROW_OFFSET = 100000;
static const uint LOCAL_PREVIOUS_ROW_OFFSET = 1024;
static const uint LOCAL_BONE_COUNT_INIPARAM = 101;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint local_row = tid.x;
    uint local_bone_count = (uint)IniParams[LOCAL_BONE_COUNT_INIPARAM].x;
    uint rows_to_copy = local_bone_count * 3;
    if (local_row >= rows_to_copy)
    {
        return;
    }

    uint local_bone = local_row / 3;
    uint row_in_bone = local_row % 3;
    uint capture_store_index = LocalPalette[local_bone];

    uint src_current_row = GLOBAL_RESERVED_ROWS + capture_store_index * 3 + row_in_bone;
    uint src_previous_row = GLOBAL_PREVIOUS_ROW_OFFSET + GLOBAL_RESERVED_ROWS + capture_store_index * 3 + row_in_bone;
    uint dst_current_row = GLOBAL_RESERVED_ROWS + local_bone * 3 + row_in_bone;
    uint dst_previous_row = LOCAL_PREVIOUS_ROW_OFFSET + GLOBAL_RESERVED_ROWS + local_bone * 3 + row_in_bone;

    LocalFakeT0_UAV[dst_current_row] = GlobalFakeT0[src_current_row];
    LocalFakeT0_UAV[dst_previous_row] = GlobalFakeT0[src_previous_row];
}
