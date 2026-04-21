# Bone Merge Capture

独立 Blender 插件，用于：

- 从目标物体列表解析 `ib_hash + match_index_count`
- 使用 `Target Collection` 集中管理要参与扫描的 IB 物体
- 在指定 `FrameAnalysis` 文件夹里找到对应 draw
- 按 draw 顺序建立全局骨序
- 生成 `capture_manifest.json`
- 生成独立 `BoneStore.ini`
- 自动导出 `hlsl/` 目录并复制捕获所需 HLSL
- 按映射表把 Blender 里的局部数字顶点组重命名为全局数字顶点组
- 单独执行“同骨顶点组”分析与权重合并

## 当前主流程

第一版主流程 **不再依赖 `vb2` 推断骨数**。

骨数来源固定为：

- 当前 Blender 物体上所有“纯数字命名”的顶点组
- `local_bone_count = max(numeric_group_name) + 1`

`FrameAnalysis` 只负责：

- 找到目标 draw
- 读取 draw 顺序
- 读取 `vs-t0` / `vs-cb1`
- 生成捕获所需的部位记录

## Target Collection

面板中的 `Target Collection` 用来集中管理“骨骼调色板需要扫描的 IB 物体”。

- `Create Target Collection`
  - 创建或选择默认集合 `BMC Bone Palette Targets`
- `Add Selected Objects`
  - 把选中的 mesh 加入目标列表
  - 同时链接到 `Target Collection`
- `Sync Targets From Collection`
  - 用集合里的 mesh 重新生成目标列表

这个集合只负责管理扫描对象；真正的 `palette.buf` 仍然等后续导出阶段再生成。

扫描完成后，插件会自动：

- 生成 `capture_manifest.json`
- 生成 `BoneStore.ini`
- 导出 `hlsl/` 子目录
- 立即对当前目标物体执行一次局部顶点组 -> 全局顶点组重命名

`palette.buf` 和本地化后的 `Blend.buf` 属于后续导出阶段，不在 `Scan and Generate` 阶段生成。

默认情况下，`Scan and Generate` 不会自动跑“同骨/接缝骨”匹配。这个空间近邻 + 权重 + 矩阵签名分析可能比较慢。

如果勾选 `Scan 后自动合并同骨顶点组`，扫描完成后会自动：

- 跑 `Analyze Same-Bone Groups`
- 把推荐 alias 写回 `capture_manifest.json`
- 立即执行一次 `Merge Duplicate Bones`

不勾选时，也可以在后处理阶段手动点击 `Analyze Same-Bone Groups` 和 `Merge Duplicate Bones`。

## Export Preparation

导出阶段使用单独的 `Export Collection`，但真正的最终绘制单位是它下面的**子集合**。

- 点击 `Create Export Collection` 会自动按当前 Target 列表创建常用子集合
- 用户也可以在 `Export Collection` 下面手动创建或重命名子集合
- 直接放在根 `Export Collection` 下的 mesh 不会被 Prepare 接受，必须移动到某个子集合里
- 子集合名称就是最终宿主 draw/chunk 身份，格式推荐为 `<ib_hash>-<match_index_count>-<chunk_index>`
  - 例如：`fe47dc61-7014-0`
  - 如果 A 的几何想挂到 B 的宿主上，就把 A 的 mesh 放到 B 对应的子集合下
- `Prepare Export Collection` 会：
  - 按子集合分组
  - 每个子集合生成一份 `Buffer/<ib>-<match_index_count>-<chunk_index>-Palette.buf`
  - 直接把子集合内 mesh 的数字顶点组从全局编号本地化成 `0..n-1`
  - 重建顶点组内部顺序，确保 3Dmigoto 导出的 `BLENDINDICES` 也变成本地连续索引
  - 对没有导出子集合的扫描 IB 生成默认原始 palette：`local i -> global_bone_base + i`
  - 写出 `export_manifest.json`

第一版不直接导出完整 3Dmigoto 网格缓冲。现有 3Dmigoto Blender 导出插件应对 `Export Collection` / 对应子集合里的 mesh 导出 `Position/Texcoord/Blend/Index` 等 mesh buffer。

注意：导出准备现在是**就地修改顶点组**，不是复制一份。插件会在物体自定义属性里保存上一次的 `palette`，下次重新 Prepare 时可以把当前本地顶点组解释回全局骨号再重新计算。如果你新增物体、移动物体到另一个子集合，或改了权重/顶点组，重新运行 `Prepare Export / Palette` 即可。

面板中的导出相关按钮在 `Export Preparation` 区域：

- `Create Export Collection`：创建导出源集合，并自动创建当前 Target 列表对应的 `<ib>-<count>-0` 子集合
- `Add Selected To Export`：把当前选中的 mesh 放入一个宿主子集合；默认使用 active mesh 的 IB 信息创建 `<ib>-<count>-0`
- `Prepare Export / Palette`：按子集合生成 `Palette.buf`，并就地本地化顶点组

## `cb1` 使用约定

- `ResourceDumpedCB1_SRV`
  - 是当前这一次 `ExtractCB1` 后的**临时 cb1 staging**
- Stage 1（source `ib` 捕获）
  - `ExtractCB1 -> RecordBones`
  - 这里只是借当前 source draw 的 `cb1[5].x/.y` 去找到原始 `vs-t0` 里的骨骼窗口并写入全局大缓冲
- Stage 2（最终 chunk 绘制）
  - 再对当前 consuming draw 重新 `ExtractCB1`
  - `RedirectCB1` 只修改这个 consuming draw 的 `cb1[5].x/.y`
  - 让它指向共享本地小缓冲区

也就是说：

- Stage 1：`ExtractCB1 -> RecordBones`
- Stage 2：`ExtractCB1 -> GatherBones -> RedirectCB1 -> bind local vs-t0/vs-cb1`

## Capture 协议

每个源 `IB` 只存两项元数据：

- `global_bone_base`
- `capture_bone_count`

运行时由 HLSL 自行计算：

- `row_base = global_bone_base * 3`

大缓冲区布局固定为：

- current: `3 + g*3 .. 3 + g*3 + 2`
- previous: `100000 + 3 + g*3 .. +2`

## 输出

- `capture_manifest.json`
- `BoneStore.ini`
- `hlsl/extract_cb1_vs.hlsl`
- `hlsl/extract_cb1_ps.hlsl`
- `hlsl/gather_bones_cs.hlsl`
- `hlsl/record_bones_dynamic_cs.hlsl`
- `hlsl/redirect_cb1_cs.hlsl`

## 备注

- 第一版只处理 `vs == 200`
- “重复骨/接缝骨合并”是后处理按钮，不影响主捕获布局
- `Analyze Same-Bone Groups` 推荐现在必须同时满足：
  - 接缝顶点在世界空间内互为最近点
  - 至少有多对接缝顶点支持这个物体对
  - 某个顶点组映射必须有多票权重支持
  - 接缝顶点上的权重值相似
  - 对应全局骨的 current/previous `vs-t0` 矩阵签名一致
- 当前工作流按 `BI4 / R8G8B8A8_UINT` 约束处理：每个目标物体/最终 draw chunk 的局部骨数必须 `<=256`
- 扫描阶段允许建立更大的全局骨序；真正需要受 `256` 限制的是单个最终绘制块，而不是 capture 阶段的全局骨总数

## Local Palette 路线（第一版协议）

为避免修改 IA / BLENDINDICES 格式，推荐后续导出器走：

- Blender 内继续保留**全局骨号**
- 导出每个最终 draw chunk 时，生成一个 `palette.buf`
- `palette[localBone] = globalBone`
- 顶点里的 `BLENDINDICES` 改写成本地连续 `0..n-1`
- 运行时推荐顺序：
  1. Stage 1（source `ib`，只在 `vs == 200` 时做一次）
     - `CustomShader_ExtractCB1`
     - `CustomShader_RecordBones`
  2. Stage 2（最终 consuming draw）
     - `CustomShader_ExtractCB1`
     - `CustomShader_GatherBones`
     - 绑定 `vs-t0 = ResourceLocalFakeT0_SRV`
     - `CustomShader_RedirectCB1`
     - 绑定 `vs-cb1 = ResourceFakeCB1`
     - 再 draw

### `palette.buf`

- 类型：`Buffer`
- 格式：`R32_UINT`
- 步长：`4 bytes`
- 含义：`localBone -> globalBone`

### `palette meta`

- 类型：`Buffer`
- 格式：`R32_FLOAT`
- 字段：
  - `x = local_bone_count`

### 行布局

全局大骨缓冲：

- `current = 3 + globalBone*3 + rowInBone`
- `previous = 100000 + 3 + globalBone*3 + rowInBone`

本地 gather 后小骨缓冲：

- `current = 3 + localBone*3 + rowInBone`
- `previous = 1024 + 3 + localBone*3 + rowInBone`

对应本地绘制时 `RedirectCB1` 改写后的 `cb1[5]`：

- `cb1[5].x = 0`
- `cb1[5].y = 1024`

这两个“骨数”是不同概念：

- `capture_bone_count`
  - 原始 source IB 的局部骨数
  - 给 `CustomShader_RecordBones` 用
- `local_bone_count`
  - 一个最终导出 chunk 的本地 palette 骨数
  - 给 `CustomShader_GatherBones` 用
