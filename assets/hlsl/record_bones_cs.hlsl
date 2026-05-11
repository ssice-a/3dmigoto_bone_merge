// Capture native bones into the canonical global bone pool.
//
// t0 = native game vs-t0
// t1 = dumped shader-visible cb1
// t2 = MainCaptureBoneMap or LodCaptureBoneMap
// u1 = GlobalBonePool
// u2 = RuntimeState
//
// x100 = capture record index

#include "bone_store_common.hlsli"

StructuredBuffer<uint4> NativeT0 : register(t0);
StructuredBuffer<uint4> DumpedCB1 : register(t1);
Buffer<uint> CaptureBoneMap : register(t2);
Texture1D<float4> IniParams : register(t120);

RWStructuredBuffer<uint4> GlobalBonePool_UAV : register(u1);
RWStructuredBuffer<uint4> RuntimeState_UAV : register(u2);

bool IsZero4(uint4 value)
{
    return all(value == uint4(0, 0, 0, 0));
}

void EnsureCaptureSlot()
{
    uint4 header = RuntimeState_UAV[BMC_STATE_HEADER];
    if (header.y == BMC_INVALID_SLOT || header.z == 0)
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

    if (tid.x == 0)
    {
        EnsureCaptureSlot();
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

    uint native_slot = 0;
    uint native_cb1_bone_row = native_slot * BMC_CB1_INSTANCE_STRIDE + BMC_CB1_BONE_BASE_ROW;
    uint dst_slot_base = slot * BMC_GLOBAL_SLOT_ROW_STRIDE;

    uint4 bone_window = DumpedCB1[native_cb1_bone_row];
    uint src_current_base = bone_window.x;
    uint src_previous_base = bone_window.y;
    uint capture_valid = 0;

    if (pair_count != 0 && !IsZero4(bone_window))
    {
        uint max_source_local_bone = 0;
        uint first_source_local_bone = 0;

        for (uint pair_index = 0; pair_index < pair_count; ++pair_index)
        {
            uint pair_offset = pair_table_uint_base + (pair_base + pair_index) * BMC_CAPTURE_PAIR_STRIDE;
            uint source_local_bone = CaptureBoneMap[pair_offset + 0];

            if (pair_index == 0)
            {
                first_source_local_bone = source_local_bone;
            }

            max_source_local_bone = max(max_source_local_bone, source_local_bone);
        }

        uint native_rows = 0;
        uint native_stride = 0;
        NativeT0.GetDimensions(native_rows, native_stride);

        uint required_current_row = src_current_base + BMC_BONE_RESERVED_ROWS + max_source_local_bone * BMC_BONE_ROWS + (BMC_BONE_ROWS - 1);
        uint required_previous_row = src_previous_base + BMC_BONE_RESERVED_ROWS + max_source_local_bone * BMC_BONE_ROWS + (BMC_BONE_ROWS - 1);

        if (required_current_row < native_rows && required_previous_row < native_rows)
        {
            uint first_current_row = src_current_base + BMC_BONE_RESERVED_ROWS + first_source_local_bone * BMC_BONE_ROWS;
            uint4 row0 = NativeT0[first_current_row + 0];
            uint4 row1 = NativeT0[first_current_row + 1];
            uint4 row2 = NativeT0[first_current_row + 2];

            if (!IsZero4(row0) || !IsZero4(row1) || !IsZero4(row2))
            {
                capture_valid = 1;
            }
        }
    }

    if (capture_valid != 0)
    {
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
    }

    if (tid.x == 0 && capture_valid != 0)
    {
        RuntimeState_UAV[BMC_STATE_SLOT_BASE + slot] = uint4(1, RuntimeState_UAV[BMC_STATE_HEADER].x, 0, 0);
    }
}
