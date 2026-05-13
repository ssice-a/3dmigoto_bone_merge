// Shared constants for Bone Merge Capture runtime shaders.

static const uint BMC_MAX_INSTANCE_SLOTS = 4;
static const uint BMC_INVALID_SLOT = 0xffffffffu;

static const uint BMC_CB1_ROW_COUNT = 4096;
static const uint BMC_CB1_INSTANCE_STRIDE = 16;
static const uint BMC_CB1_BONE_BASE_ROW = 5;

static const uint BMC_BONE_RESERVED_ROWS = 3;
static const uint BMC_BONE_ROWS = 3;

static const uint BMC_GLOBAL_PREVIOUS_ROW_OFFSET = 100000;
static const uint BMC_GLOBAL_SLOT_ROW_STRIDE = 200000;

static const uint BMC_LOCAL_PREVIOUS_ROW_OFFSET = 1024;
static const uint BMC_LOCAL_SLOT_ROW_STRIDE = 2048;

static const uint BMC_LOCAL_BONE_COUNT_INIPARAM = 101;

static const uint BMC_CAPTURE_MAP_HEADER_UINTS = 4;
static const uint BMC_CAPTURE_PAIR_STRIDE = 2;

static const uint BMC_STATE_HEADER = 0;
static const uint BMC_STATE_SLOT_BASE = 1;
static const uint BMC_STATE_OVERFLOW_FLAG = 1;

#define BMC_INI_PARAM_UINT(ini_params, index) ((uint)((ini_params)[(index)].x + 0.5))
