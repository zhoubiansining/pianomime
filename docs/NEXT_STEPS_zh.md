# 后续工作提纲

最后更新：2026-05-27

baseline 复现和报告展示材料已经准备完成。接下来建议按“报告合入、展示整理、改进实验”三条线推进。

## 1. 报告合入

1. 把 `docs/BASELINE_REPORT_SECTION.tex` 合入最终 report 的 experiment/baseline subsection。
2. 确认主 tex 文件包含 `graphicx` 和 `booktabs`。
3. 根据最终 report 的图片目录，调整 `\includegraphics` 中的路径。
4. 将 `docs/report_assets/ppo_petrunko_f1_curve.png` 放入正文图。
5. 将 single-song 和 generalist baseline 表格保留在正文；如果篇幅不够，generalist 全表可放 appendix。

## 2. 展示材料

1. 从 `docs/report_assets/` 中选择 3-5 张截图放入 slides/poster。
2. 视频展示建议使用：
   - `Stan_1_single_song_baseline.mp4`
   - `Petrunko_3_single_song_baseline.mp4`
   - `NoTimeToDie_1_multisong_baseline.mp4`
   - `EyesClosed_1_multisong_baseline.mp4`
   - `Paradise_1_multisong_baseline.mp4`
3. 展示时说明当前视频是 silent video，视觉仿真和 note-level metrics 有效。
4. 如果最终展示必须有声音，需要额外安装系统级 FluidSynth/PortAudio，并重新渲染选定视频。

## 3. 改进实验

1. Single-song improvement：
   - 选择一首较难或较长的 training-set song。
   - 复制 `configs/baseline.toml`，新建方法配置。
   - 与当前 baseline 对比 F1 curve 和 play-through video。
2. Generalist improvement：
   - 至少选择 5 首 unseen/test songs。
   - 推荐先用 `NoTimeToDie_1`、`EyesClosed_1`、`Paradise_1`、`Forester_1`、`Alone_1`。
   - 报告 baseline 与 improved F1，并提供视频对比。
3. 结果管理：
   - 不覆盖 `/home/gaoj/share4/_piano/baseline_results`。
   - 新方法结果建议写到 `improvement_results/` 或带 method/run_id 的新目录。
   - 每次实验记录 config、command、git commit、seed、song、metrics、logs、video。

## 4. 仓库维护

1. 本地修改经检查通过后，再 push 到 GitHub。
2. 当前 GitHub 仓库包含 `dataset_hl.zarr` 和 `dataset_ll.zarr`，方便复现但增加仓库体积；如果后续仓库变慢，建议迁回 artifact cache/download 方案。
3. 新增实验入口时同步更新 `configs/`、`docs/USAGE_zh.md` 和相关结果文档。
