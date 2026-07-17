// Shared constants for the EFMI keyed-instance bone runtime.

static const uint BMC_MAX_INSTANCE_SLOTS = 8;
static const uint BMC_INVALID_SLOT = 0xffffffffu;

static const uint BMC_CB_ROW_COUNT = 4096;
static const uint BMC_CB_INSTANCE_STRIDE = 16;
static const uint BMC_CB_BONE_BASE_ROW = 5;

static const uint BMC_BONE_RESERVED_ROWS = 3;
static const uint BMC_BONE_ROWS = 3;

static const uint BMC_MAX_GLOBAL_BONES = 4096;
static const uint BMC_GLOBAL_PREVIOUS_ROW_OFFSET = BMC_BONE_RESERVED_ROWS + BMC_MAX_GLOBAL_BONES * BMC_BONE_ROWS;
static const uint BMC_GLOBAL_SLOT_ROW_STRIDE = BMC_GLOBAL_PREVIOUS_ROW_OFFSET * 2;

static const uint BMC_MAX_LOCAL_BONES = 256;
static const uint BMC_LOCAL_PREVIOUS_ROW_OFFSET = BMC_BONE_RESERVED_ROWS + BMC_MAX_LOCAL_BONES * BMC_BONE_ROWS;
static const uint BMC_LOCAL_SLOT_ROW_STRIDE = BMC_LOCAL_PREVIOUS_ROW_OFFSET * 2;

static const uint BMC_CAPTURE_MAP_HEADER_UINTS = 4;
static const uint BMC_CAPTURE_PAIR_STRIDE = 2;

static const uint BMC_STATE_HEADER = 0;
static const uint BMC_STATE_SLOT_BASE = 1;
