// Gather one part-local palette through native-slot -> capture-slot mapping.
//
// t0 = GlobalBonePool
// t1 = InstanceMapping
// t2 = PartLocalToGlobalBoneMap, uint[0] = local bone count, uint[1..] = local -> global
// u1 = LocalBonePool
// u2 = RuntimeState

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> GlobalBonePool : register(t0);
StructuredBuffer<uint4> InstanceMapping : register(t1);
Buffer<uint> PartLocalToGlobalBoneMap : register(t2);

RWStructuredBuffer<uint4> LocalBonePool_UAV : register(u1);
RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint local_bone_count = PartLocalToGlobalBoneMap[0];
    if (local_bone_count == 0 || local_bone_count > BMC_MAX_LOCAL_BONES)
    {
        return;
    }

    uint rows_to_copy = local_bone_count * BMC_BONE_ROWS;

    for (uint native_slot = 0; native_slot < BMC_MAX_INSTANCE_SLOTS; ++native_slot)
    {
        uint4 mapping = InstanceMapping[native_slot];
        uint capture_slot = mapping.x;
        if (mapping.y == 0 || capture_slot >= BMC_MAX_INSTANCE_SLOTS)
        {
            continue;
        }

        uint4 slot_state = RuntimeState_UAV[BMC_STATE_SLOT_BASE + capture_slot];
        if (slot_state.x == 0)
        {
            continue;
        }

        uint src_slot_base = capture_slot * BMC_GLOBAL_SLOT_ROW_STRIDE;
        uint dst_slot_base = native_slot * BMC_LOCAL_SLOT_ROW_STRIDE;

        for (uint local_row = tid.x; local_row < rows_to_copy; local_row += 32)
        {
            uint local_bone = local_row / BMC_BONE_ROWS;
            uint row_in_bone = local_row % BMC_BONE_ROWS;
            uint global_bone = PartLocalToGlobalBoneMap[1 + local_bone];
            if (global_bone >= BMC_MAX_GLOBAL_BONES)
            {
                continue;
            }

            uint src_current_row = src_slot_base + BMC_BONE_RESERVED_ROWS + global_bone * BMC_BONE_ROWS + row_in_bone;
            uint src_previous_row = src_slot_base + BMC_GLOBAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + global_bone * BMC_BONE_ROWS + row_in_bone;
            uint dst_current_row = dst_slot_base + BMC_BONE_RESERVED_ROWS + local_bone * BMC_BONE_ROWS + row_in_bone;
            uint dst_previous_row = dst_slot_base + BMC_LOCAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + local_bone * BMC_BONE_ROWS + row_in_bone;

            LocalBonePool_UAV[dst_current_row] = GlobalBonePool[src_current_row];
            LocalBonePool_UAV[dst_previous_row] = GlobalBonePool[src_previous_row];
        }
    }
}
