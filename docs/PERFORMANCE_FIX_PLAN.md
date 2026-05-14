# BMC Performance Fix Plan

本文记录导入、导出、Apply Pool、LOD 分析的性能热点，以及后续修复顺序。目标是让热路径明确走 numpy/批量 API，删掉旧循环路径和重复建表，同时不改变运行时 INI 的三阶段语义。

## 当前现象

近期日志里主要有三类慢点：

- `Apply Pool`: `seam_merge` 可以占到 12s 以上，主要耗在接缝 cache、顶点组点云、候选配对。
- `Export`: `geometry` 可以占到 6s 到 8s，热点集中在 `collect_loops`、`vb0:Position`、`vb2:Blend`。
- `LOD Analyze`: 点云匹配本身局部使用了 numpy，但建云、采样、候选诊断和 manifest 写入仍有明显对象/字典开销。

另一个独立问题是 manifest 体积偏大。`lxi` 的 `capture_manifest.json` 约 1MB，其中大量是诊断数据，不该全部进入每次导出读取的运行路径。

## 硬约束

- numpy 可用时，热路径必须使用 numpy；不要保留旧循环 fallback 来增加维护成本。
- INI 运行规则不因性能修复改变：record 阶段、shadow 延迟绘制阶段、主上色阶段仍然分开。
- 主控和 LOD 的 capture/replay 语义由导出计划决定，性能优化不能重新引入链首/链尾猜 profile 的逻辑。
- 输出 buffer 的字节结果要可比较。性能改动必须用 hash 或二进制比较确认没有改变 `Position`、`Texcoord`、`Blend`、`Index`。

## 热点 1: 导出几何 buffer

相关函数：

- `core/export_buffers.py::_collect_part_loop_vertices`
- `core/export_buffers.py::_collect_mesh_loop_vertices_fast`
- `core/export_buffers.py::_write_numpy_position_slot`
- `core/export_buffers.py::_write_numpy_blend_slot`
- `core/export_buffers.py::_loop_vertex_mesh_range_arrays`
- `core/export_buffers.py::_local_top4_vertex_arrays`

目前即使走了 fast path，仍会为每个 mesh loop 创建 `_LoopVertex` Python 对象。后续 `vb0`、`vb1`、`vb2` 又从这个对象列表里提取 `vertex_index`、`loop_index`、mesh range。几十万 loops 时，这会把 numpy 路径前面又塞回 Python 对象层。

修复方向：

- 引入 `ExportLoopBatch` 或类似结构，直接保存数组：
  - `object_ranges`
  - `vertex_indices`
  - `loop_indices`
  - `index_array`
  - 对应 mesh/cache 引用
- `_collect_part_loop_vertices` 不再返回 `_LoopVertex` 列表作为主数据结构，只保留对象 draw range 和数组。
- `vb0` 写入直接按 batch 取坐标、法线、切线、mirror/transform 后一次性写入。
- `vb2` 写入直接按 batch 取 top4 权重索引数组，避免从 `_LoopVertex` 反查。
- `_local_top4_vertex_arrays` 的 cache key 必须包含 palette/local remap 身份。它不是纯 mesh 数据，不能只按 mesh 缓存。
- `vb3` 已 alias 到 `vb0`，继续避免重复写。

优先级最高，因为当前导出日志里 `geometry` 是最稳定的大头。

## 热点 2: Apply Pool 接缝匹配

相关函数：

- `core/seam_matcher.py::build_and_apply_seam_mapping`
- `core/seam_matcher.py::_build_seam_cache`
- `core/seam_matcher.py::_collect_seam_vertices_numpy`
- `core/seam_matcher.py::_build_sorted_vertex_weight_cache`
- `core/seam_matcher.py::_read_vertex_weights`
- `core/seam_matcher.py::_build_group_clouds`

现状是边界顶点收集已经用 `foreach_get`，但权重读取和 group cloud 构建仍然容易回到 Blender 顶点组逐点循环。日志里 `group_clouds`、`cache_build`、`allowed_groups` 都曾经是大头。

修复方向：

- seam cache 按 mesh/object 状态缓存，缓存 key 至少包含：
  - object name/data pointer
  - vertex/edge/loop 数量
  - vertex group 名称集合
  - mesh 修改标记或保守地在 Apply Pool 按钮级别缓存一次
- 将 group cloud 生成改成数组聚合：
  - 边界 vertex indices 数组
  - world coords 数组
  - group id/weight 扁平数组
  - 使用 sort/reduceat 或 bincount 聚合 bounds、cell keys、候选点
- `_build_sorted_vertex_weight_cache` 保留 API 层不可避免的读取，但结果应批量缓存，避免同一次 Apply Pool 内重复读。
- 选中对象、manifest、顶点组命名未变化时，不重复 seam_merge。

这部分受 Blender vertex group API 限制，不能保证完全无循环，但可以把循环压缩到一次读取和一次缓存。

## 热点 3: LOD 分析和点云匹配

相关函数：

- `core/lod_analyze.py::analyze_lod_for_manifest`
- `core/lod_analyze.py::build_lod_bone_cloud_mapping`
- `core/lod_analyze.py::_sample_canonical_bone_clouds`
- `core/lod_analyze.py::_sample_lod_bone_clouds`
- `core/lod_analyze.py::_score_bone_clouds`
- `core/lod_analyze.py::_load_candidate_point_geometry`

`_score_bone_clouds` 已经有 numpy 网格和距离计算，但前后的 `WeightedPoint`、`BoneSample`、候选字典、诊断记录仍然很重。当前 manifest 中 `lod_capture_records`、`lod_mapping`、`lod_manifest_snapshot`、`texture_candidates`、`draw_hits` 都很大。

修复方向：

- 对点云输入建立持久化 `.npz` cache：
  - key 使用 frameanalysis 路径、dump 文件大小/mtime、IB hash/index count、VB layout fingerprint。
  - 缓存 canonical cloud 和 lod cloud 的数组版本。
- 同 IB 且主控/LOD 映射表等价时直接复用映射，不再重新点云匹配。
- 对 `build_lod_bone_cloud_mapping` 增加数组输入路径，减少 `BoneSample` 对象。
- 候选裁剪先用 bounds/centroid/点数，只有通过粗筛才做详细 sample scoring。
- 缺失 LOD 骨骼时优先做确定性补全：
  - 同一 LOD part 内 donor link 补全。
  - 同 hash/同 index count/同 vb2 signature 复用。
  - `no capture` 和 `DYNAMIC_VB0` 类记录保持 ignored，不参与阻塞。

## 热点 4: 导入物体

相关函数：

- `core/import_candidates.py::load_candidate_geometry`
- `core/import_candidates.py::create_blender_object_from_geometry`
- `core/import_candidates.py::_read_numpy_vector_records`
- `core/import_candidates.py::_apply_uv_layer_numpy`
- `core/import_candidates.py::_assign_vertex_groups`

导入读 VB/IB 已有 numpy 辅助，但 Blender 写 mesh、UV、attribute、vertex group 仍可能逐点操作。

修复方向：

- 读 buffer 使用 structured numpy view，按语义一次性切片。
- 写 mesh 使用 `foreach_set`，避免逐顶点写坐标、UV、属性。
- 顶点组写入按 group 聚合：
  - 先从 blend indices/weights 生成 `group -> vertex_indices, weights`。
  - 同权重或近似同权重批量 `group.add(indices, weight, 'ADD')`。
  - 如果权重必须逐点，至少只在最终 Blender API 层循环。
- header、slot slice、semantic 解析按文件 path/mtime/size 缓存。

## 热点 5: JSON 和 manifest

相关函数：

- `core/io.py::write_json`
- `core/main_analyze.py`
- `core/export_prepare.py`
- `core/ini_export.py`

`write_json` 当前固定 `indent=2`。对调试友好，但运行路径频繁读写时没有必要。`capture_manifest.json` 同时承载运行所需数据和大量调试信息，导致体积膨胀。

修复方向：

- 给 `write_json` 增加 `compact` 参数：
  - 运行 manifest 使用 `separators=(",", ":")`。
  - 手工调试 manifest 可以继续 pretty print。
- 拆分 manifest：
  - `capture_manifest.json`: 运行最小数据。
  - `capture_manifest.debug.json`: 点云候选、draw dump 路径、完整 snapshot、warnings 详情。
  - `export_manifest.json`: INI 生成需要的数据。
  - `export_manifest.debug.json`: geometry timings、runtime 展开、诊断。
- 从运行 manifest 移除或压缩：
  - `lod_manifest_snapshot.candidate_ibs`
  - 大量 `draw_hits` dump path
  - `texture_candidates` 的完整记录
  - `global_candidates` 全量候选，只保留 selected 和 warning 摘要

JSON 不是当前最大 CPU 热点，但这是低风险收益，能减少磁盘 IO、diff 噪音和反复读取成本。

## 建议实施顺序

1. 增加细粒度计时，不改行为：
   - export: `collect_loop_arrays`、`top4_arrays`、`position_pack`、`blend_pack`、`file_write`
   - seam: `boundary_vertices`、`weight_read`、`group_clouds`、`allowed_pairs`、`nearest_pairs`
   - lod: `load_geometry`、`point_cloud_build`、`bone_cloud_sample`、`score`、`capture_records`
   - import: `read_buffers`、`mesh_create`、`uv_write`、`group_assign`
2. 先修 JSON：
   - 增加 compact 写入。
   - 拆 debug manifest。
3. 重写导出几何主路径：
   - `_LoopVertex` 列表退居兼容测试用途，热路径改数组 batch。
   - `vb0/vb2` 直接吃数组。
4. 优化 Apply Pool：
   - 缓存 seam cache。
   - 数组化 group cloud。
5. 优化 LOD 分析：
   - `.npz` 点云缓存。
   - 同 IB/同 vb2 signature 复用映射。
   - 缩减候选诊断。
6. 优化导入：
   - buffer read cache。
   - vertex group 批量写入。

## 验证方式

- 对同一 Blender 场景导出前后比较所有 `.buf` 文件 hash。
- 对同一 frameanalysis 运行 LOD Analyze，比较 selected mapping 和 capture records。
- 对 Apply Pool 的结果比较顶点组重命名和 alias 数量。
- 保留性能日志，至少记录：
  - Apply Pool 总耗时和 seam_merge 耗时。
  - Export 总耗时、geometry 耗时、slot 耗时。
  - LOD Analyze 总耗时、point cloud/scoring 耗时。
  - Import 总耗时、mesh/group 写入耗时。

## 当前优先修复点

第一优先级是导出几何数组化。它直接解释了 `collect_loops` 和 `vb0/vb2` 的高耗时，并且风险可控，因为输出 buffer 可以逐字节比较。

第二优先级是 seam cache/group cloud。Apply Pool 的 12s 级耗时主要在这里。

第三优先级是 LOD 点云持久缓存和 manifest 精简。它能降低反复分析 LOD 的等待时间，也能让后续问题定位更清楚。

## 已实施

- `write_json(..., compact=True)` 已加入，主分析 `capture_manifest.json` 和导出 `export_manifest.json` 现在走 compact 写入，减少 manifest 体积和写盘时间。
- 导出几何 fast path 已避免为每个 loop 创建 `_LoopVertex` 对象。正常 Blender mesh 只保留每个 mesh range 的代表对象，并用缓存的 `vertex_indices`/`loop_indices` 数组供 `vb0/vb1/vb2` 写入使用。
- R32 index buffer 写入增加 numpy 打包路径，list/ndarray 输入不再经过 Python 循环加 `array("I")` 的二次打包。

## 仍待实施

- `vb0/vb2` 继续向真正的 batch writer 迁移，最终让 slot writer 直接接收 loop batch 数组，而不是依赖代表 `_LoopVertex`。
- Apply Pool 的 group cloud 构建仍需要数组化和缓存。
- LOD 点云需要 `.npz` 持久缓存，并拆出 debug manifest。
