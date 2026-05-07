// Capture native bones into the canonical global bone pool.
//
// t0 = native game vs-t0
// t1 = dumped shader-visible cb1
// t2 = MainCaptureBoneMap or LodCaptureBoneMap
// u1 = GlobalBonePool
// u2 = RuntimeState
//
// x100 = capture record index
// x102 = capture sequence flags: bit0 begin, bit1 end
// x103 = native capture instance slot, defaults to 0

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> NativeT0 : register(t0);
StructuredBuffer<uint4> DumpedCB1 : register(t1);
Buffer<uint> CaptureBoneMap : register(t2);
Texture1D<float4> IniParams : register(t120);

RWStructuredBuffer<uint4> GlobalBonePool_UAV : register(u1);
RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);

void EnsureCaptureSlot(uint flags)
{
    uint4 header = RuntimeState_UAV[BMC_STATE_HEADER];
    uint active_slot = header.y;

    if ((flags & BMC_CAPTURE_SEQUENCE_BEGIN) != 0)
    {
        uint next_slot = (active_slot == BMC_INVALID_SLOT || header.z == 0) ? 0 : active_slot + 1;
        if (next_slot < BMC_MAX_INSTANCE_SLOTS)
        {
            header.y = next_slot;
            header.z = max(header.z, next_slot + 1);
            RuntimeState_UAV[BMC_STATE_HEADER] = header;
            RuntimeState_UAV[BMC_STATE_SLOT_BASE + next_slot] = uint4(0, header.x, 0, 0);
        }
        else
        {
            header.w |= BMC_STATE_OVERFLOW_FLAG;
            RuntimeState_UAV[BMC_STATE_HEADER] = header;
        }
        return;
    }

    if (active_slot == BMC_INVALID_SLOT || header.z == 0)
    {
        header.y = 0;
        header.z = max(header.z, 1);
        RuntimeState_UAV[BMC_STATE_HEADER] = header;
        RuntimeState_UAV[BMC_STATE_SLOT_BASE] = uint4(0, header.x, 0, 0);
    }
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint record_index = BMC_INI_PARAM_UINT(IniParams, BMC_CAPTURE_RECORD_INIPARAM);
    uint flags = BMC_INI_PARAM_UINT(IniParams, BMC_CAPTURE_SEQUENCE_FLAGS_INIPARAM);

    if (tid.x == 0)
    {
        EnsureCaptureSlot(flags);
    }
    GroupMemoryBarrierWithGroupSync();

    uint record_count = CaptureBoneMap[0];
    if (record_index >= record_count)
    {
        return;
    }

    uint slot = RuntimeState_UAV[BMC_STATE_HEADER].y;
    if (slot == BMC_INVALID_SLOT || slot >= BMC_MAX_INSTANCE_SLOTS)
    {
        return;
    }

    uint pair_table_uint_base = CaptureBoneMap[1];
    uint record_base = BMC_CAPTURE_MAP_HEADER_UINTS + record_index * BMC_CAPTURE_RECORD_STRIDE;
    uint pair_base = CaptureBoneMap[record_base + 0];
    uint pair_count = CaptureBoneMap[record_base + 1];
    uint rows_to_copy = pair_count * BMC_BONE_ROWS;

    uint native_slot = BMC_INI_PARAM_UINT(IniParams, BMC_NATIVE_CAPTURE_INSTANCE_SLOT_INIPARAM);
    uint native_cb1_bone_row = native_slot * BMC_CB1_INSTANCE_STRIDE + BMC_CB1_BONE_BASE_ROW;
    uint src_current_base = DumpedCB1[native_cb1_bone_row].x;
    uint src_previous_base = DumpedCB1[native_cb1_bone_row].y;
    uint dst_slot_base = slot * BMC_GLOBAL_SLOT_ROW_STRIDE;

    for (uint pair_row = tid.x; pair_row < rows_to_copy; pair_row += 64)
    {
        uint pair_index = pair_row / BMC_BONE_ROWS;
        uint row_in_bone = pair_row % BMC_BONE_ROWS;
        uint pair_offset = pair_table_uint_base + (pair_base + pair_index) * BMC_CAPTURE_PAIR_STRIDE;
        uint source_local_bone = CaptureBoneMap[pair_offset + 0];
        uint target_global_bone = CaptureBoneMap[pair_offset + 1];

        uint src_current_row = src_current_base + BMC_BONE_RESERVED_ROWS + source_local_bone * BMC_BONE_ROWS + row_in_bone;
        uint src_previous_row = src_previous_base + BMC_BONE_RESERVED_ROWS + source_local_bone * BMC_BONE_ROWS + row_in_bone;
        uint dst_current_row = dst_slot_base + BMC_BONE_RESERVED_ROWS + target_global_bone * BMC_BONE_ROWS + row_in_bone;
        uint dst_previous_row = dst_slot_base + BMC_GLOBAL_PREVIOUS_ROW_OFFSET + BMC_BONE_RESERVED_ROWS + target_global_bone * BMC_BONE_ROWS + row_in_bone;

        GlobalBonePool_UAV[dst_current_row] = NativeT0[src_current_row];
        GlobalBonePool_UAV[dst_previous_row] = NativeT0[src_previous_row];
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid.x == 0 && (((flags & BMC_CAPTURE_SEQUENCE_END) != 0) || flags == 0))
    {
        RuntimeState_UAV[BMC_STATE_SLOT_BASE + slot] = uint4(1, RuntimeState_UAV[BMC_STATE_HEADER].x, 0, 0);
    }
}
