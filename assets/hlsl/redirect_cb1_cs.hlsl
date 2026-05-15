// Redirect cb1 bone-window rows to the local per-instance bone pool.
//
// t0 = dumped shader-visible cb1
// u0 = fake cb1
// u2 = RuntimeState

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> DumpedCB1 : register(t0);

RWStructuredBuffer<uint4> FakeCB1_UAV : register(u0);
RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    for (uint id = tid.x; id < BMC_CB1_ROW_COUNT; id += 32)
    {
        uint4 cb_data = DumpedCB1[id];
        uint active_slots = RuntimeState_UAV[BMC_STATE_HEADER].z;
        active_slots = min(max(active_slots, 1), BMC_MAX_INSTANCE_SLOTS);

        for (uint slot = 0; slot < active_slots; ++slot)
        {
            uint bone_window_row = slot * BMC_CB1_INSTANCE_STRIDE + BMC_CB1_BONE_BASE_ROW;
            if (id == bone_window_row)
            {
                cb_data.x = slot * BMC_LOCAL_SLOT_ROW_STRIDE;
                cb_data.y = slot * BMC_LOCAL_SLOT_ROW_STRIDE + BMC_LOCAL_PREVIOUS_ROW_OFFSET;
            }
        }

        FakeCB1_UAV[id] = cb_data;
    }
}
