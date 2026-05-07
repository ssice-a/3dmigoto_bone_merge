// Gather one part-local palette for every active instance slot.
//
// t0 = GlobalBonePool
// t2 = PartLocalToGlobalBoneMap
// u1 = LocalBonePool
// u2 = RuntimeState
//
// x101 = part local bone count

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> GlobalBonePool : register(t0);
Buffer<uint> PartLocalToGlobalBoneMap : register(t2);
Texture1D<float4> IniParams : register(t120);

RWStructuredBuffer<uint4> LocalBonePool_UAV : register(u1);
RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint local_bone_count = BMC_INI_PARAM_UINT(IniParams, BMC_LOCAL_BONE_COUNT_INIPARAM);
    if (local_bone_count == 0)
    {
        return;
    }

    uint active_slots = RuntimeState_UAV[BMC_STATE_HEADER].z;
    active_slots = min(max(active_slots, 1), BMC_MAX_INSTANCE_SLOTS);
    uint rows_to_copy = local_bone_count * BMC_BONE_ROWS;

    for (uint slot = 0; slot < active_slots; ++slot)
    {
        uint4 slot_state = RuntimeState_UAV[BMC_STATE_SLOT_BASE + slot];
        if (slot_state.x == 0)
        {
            continue;
        }

        uint src_slot_base = slot * BMC_GLOBAL_SLOT_ROW_STRIDE;
        uint dst_slot_base = slot * BMC_LOCAL_SLOT_ROW_STRIDE;

        for (uint local_row = tid.x; local_row < rows_to_copy; local_row += 64)
        {
            uint local_bone = local_row / BMC_BONE_ROWS;
            uint row_in_bone = local_row % BMC_BONE_ROWS;
            uint global_bone = PartLocalToGlobalBoneMap[local_bone];

            uint src_current_row = src_slot_base + BMC_BONE_RESERVED_ROWS + global_bone * BMC_BONE_ROWS + row_in_bone;
            uint src_previous_row = src_slot_base + BMC_GLOBAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + global_bone * BMC_BONE_ROWS + row_in_bone;
            uint dst_current_row = dst_slot_base + BMC_BONE_RESERVED_ROWS + local_bone * BMC_BONE_ROWS + row_in_bone;
            uint dst_previous_row = dst_slot_base + BMC_LOCAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + local_bone * BMC_BONE_ROWS + row_in_bone;

            LocalBonePool_UAV[dst_current_row] = GlobalBonePool[src_current_row];
            LocalBonePool_UAV[dst_previous_row] = GlobalBonePool[src_previous_row];
        }
    }
}
