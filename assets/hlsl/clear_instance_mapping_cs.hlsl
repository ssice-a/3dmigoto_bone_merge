// Clear draw-local native instance mappings without invalidating captured bones.

#include "bone_store_common.hlsli"

RWStructuredBuffer<uint4> InstanceMapping_UAV : register(u3);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    for (uint slot = tid.x; slot < BMC_MAX_INSTANCE_SLOTS; slot += 32)
    {
        InstanceMapping_UAV[slot] = uint4(BMC_INVALID_SLOT, 0, 0, 0);
    }
}
