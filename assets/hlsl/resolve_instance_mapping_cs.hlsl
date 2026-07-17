// Resolve a native game instance slot to the FIFO capture slot selected by INI.

#include "bone_store_common.hlsli"

RWStructuredBuffer<uint4> InstanceMapping_UAV : register(u3);
Texture1D<float4> IniParams : register(t120);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint capture_slot = (uint)IniParams[0].x;
    uint native_slot = (uint)IniParams[0].y;
    if (capture_slot >= BMC_MAX_INSTANCE_SLOTS || native_slot >= BMC_MAX_INSTANCE_SLOTS)
    {
        return;
    }

    InstanceMapping_UAV[native_slot] = uint4(capture_slot, 1, 0, 0);
}
