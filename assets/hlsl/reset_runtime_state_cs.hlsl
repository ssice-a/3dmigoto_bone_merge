// Clear per-frame runtime state. Matrix rows are left untouched; validity
// flags prevent stale data from being consumed next frame.

#include "bone_store_common.hlsli"

RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);
RWStructuredBuffer<uint4> InstanceMapping_UAV : register(u3);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0)
    {
        uint next_frame_id = RuntimeState_UAV[BMC_STATE_HEADER].x + 1;
        RuntimeState_UAV[BMC_STATE_HEADER] = uint4(next_frame_id, 0, 0, 0);
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint slot = tid.x; slot < BMC_MAX_INSTANCE_SLOTS; slot += 32)
    {
        RuntimeState_UAV[BMC_STATE_SLOT_BASE + slot] = uint4(0, 0, 0, 0);
        InstanceMapping_UAV[slot] = uint4(BMC_INVALID_SLOT, 0, 0, 0);
    }
}
