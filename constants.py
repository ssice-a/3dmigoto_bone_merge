"""Constants shared across the Bone Merge Capture plugin."""

BONESTORE_INI_FILE_NAME = "BoneStore.ini"
CAPTURE_MANIFEST_FILE_NAME = "capture_manifest.json"
LOD_CAPTURE_MANIFEST_FILE_NAME = "lod_capture_manifest.json"
EXPORT_MANIFEST_FILE_NAME = "export_manifest.json"
HLSL_EXPORT_DIR_NAME = "hlsl"
BUFFER_EXPORT_DIR_NAME = "Buffer"
DEFAULT_PREVIOUS_ROW_OFFSET = 100000
LOCAL_PREVIOUS_ROW_OFFSET = 1024
GLOBAL_RESERVED_ROWS = 3
FLOAT_WEIGHT_EPSILON = 1e-8
BI4_MAX_BONE_INDEX = 255
BI4_MAX_BONE_COUNT = BI4_MAX_BONE_INDEX + 1
BMC_EXPORT_PALETTE_PROP = "bmc_export_palette_values"
BMC_EXPORT_CHUNK_PROP = "bmc_export_chunk"
BMC_GLOBAL_REMAP_PROP = "bmc_global_group_remap"
BMC_GLOBAL_POOL_GENERATION_PROP = "bmc_global_pool_generation"
BMC_GLOBAL_SOURCE_KEY_PROP = "bmc_global_source_key"
BMC_ORIGINAL_LOCAL_BONE_COUNT_PROP = "bmc_original_local_bone_count"
BMC_VERTEX_GROUP_STATE_PROP = "bmc_vertex_group_state"
BMC_VERTEX_GROUP_STATE_GLOBAL = "global"
BMC_VERTEX_GROUP_STATE_EXPORT_LOCAL = "export_local"
BMC_TEXTURE_MARKS_PROP = "bmc_texture_marks_json"
BMC_TEXTURE_SLOTS_PROP = "bmc_texture_slots"
BMC_TOGGLE_DRAW_SETS_PROP = "bmc_toggle_draw_sets_json"
