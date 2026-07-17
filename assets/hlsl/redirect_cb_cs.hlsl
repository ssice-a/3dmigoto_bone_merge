// Copy the shader-visible CB and patch only bone-window offsets for mapped native slots.

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> CapturedCB : register(t0);
StructuredBuffer<uint4> InstanceMapping : register(t1);
RWStructuredBuffer<uint4> FakeCB_UAV : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    for (uint id = tid.x; id < BMC_CB_ROW_COUNT; id += 32)
    {
        uint4 cb_data = CapturedCB[id];

        for (uint native_slot = 0; native_slot < BMC_MAX_INSTANCE_SLOTS; ++native_slot)
        {
            uint4 mapping = InstanceMapping[native_slot];
            uint bone_window_row = native_slot * BMC_CB_INSTANCE_STRIDE + BMC_CB_BONE_BASE_ROW;
            if (mapping.y != 0 && id == bone_window_row)
            {
                uint local_base = native_slot * BMC_LOCAL_SLOT_ROW_STRIDE;
                cb_data.x = local_base;
                cb_data.y = local_base + BMC_LOCAL_PREVIOUS_ROW_OFFSET;
            }
        }

        FakeCB_UAV[id] = cb_data;
    }
}
