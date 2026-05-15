// Gather one part-local palette for every active instance slot.
//
// t0 = GlobalBonePool
// t2 = PartLocalToGlobalBoneMap, uint[0] = local bone count, uint[1..] = local -> global
// u1 = LocalBonePool
// u2 = RuntimeState

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> GlobalBonePool : register(t0);
Buffer<uint> PartLocalToGlobalBoneMap : register(t2);

RWStructuredBuffer<uint4> LocalBonePool_UAV : register(u1);
RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint local_bone_count = PartLocalToGlobalBoneMap[0];
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

        for (uint local_row = tid.x; local_row < rows_to_copy; local_row += 32)
        {
            uint local_bone = local_row / BMC_BONE_ROWS;
            uint row_in_bone = local_row % BMC_BONE_ROWS;
            uint global_bone = PartLocalToGlobalBoneMap[1 + local_bone];

            uint src_current_row = src_slot_base + BMC_BONE_RESERVED_ROWS + global_bone * BMC_BONE_ROWS + row_in_bone;
            uint src_previous_row = src_slot_base + BMC_GLOBAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + global_bone * BMC_BONE_ROWS + row_in_bone;
            uint dst_current_row = dst_slot_base + BMC_BONE_RESERVED_ROWS + local_bone * BMC_BONE_ROWS + row_in_bone;
            uint dst_previous_row = dst_slot_base + BMC_LOCAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + local_bone * BMC_BONE_ROWS + row_in_bone;

            LocalBonePool_UAV[dst_current_row] = GlobalBonePool[src_current_row];
            LocalBonePool_UAV[dst_previous_row] = GlobalBonePool[src_previous_row];
        }
    }
}
