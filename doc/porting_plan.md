# PARC 移植计划：develop（G1 + future-pose 蒸馏）→ upstream main（major rewrite）

> 版本：草案 v1（2026-10-14）
> 依据：6 个并行只读分析 agent 的静态分析，加上协调者对关键结论的抽查。标 ✔ 的条目已经在代码里核实过。
> 限制：目前没有任何运行时验证，因为本机没有 IsaacGym 和数据集。

## 当前进度

| 项目 | 状态 |
|---|---|
| 6 个移植分析（基线 / 角色模型与数据 / tracker env / 学习栈 / MDM 与流水线 / CVAE） | ✅ 完成 |
| uv 环境迁移（`feature/uv-env`） | ✅ 已合入 develop，CLAUDE.md 也已提交（`8385a15`），都已 push。develop 现在冻结，移植期间不动 |
| IsaacGym Preview 4 | ✅ 从 NVIDIA 官方 CDN 下载，解压到 `../isaacgym`，通过 `.pth` 链接进各自的 `.venv`。在 develop（torch 1.13）和 develop_migration（torch 2.2）上都实测通过；develop 上还跑通了 civilization 人形 tracker 训练冒烟 |
| §8 决策 D1–D12 | ✅ 已定（见 §8） |
| P0.1–P0.3（建分支、改名、打包卫生、main 的 uv 环境） | ✅ 已在 `develop_migration` 上完成 |
| 第 1 波：P0 剩余部分、P1a、P1b、Lane B | 🔄 进行中（§6） |
| 第 2 波：Lane A、C、D | ⏳ 等第 1 波合入 |
| P1c、G3 的 A/B 对比和 checkpoint 兼容检查 | ⏸ 暂缓：还没有 G1 数据集和 C 代 checkpoint（R2、R3） |

---

## 0. 摘要

- **规模**
  - develop：在 merge-base `73c79a9` 之上有 53 个非 merge 提交，内容是 Unitree G1 支持和 future-pose 师生蒸馏。
  - origin/main：upstream 的 major rewrite（`d1096c7`），改了 214 个文件。
  - 两边都改过的文件有 33 个。其中 9 个在 main 上被删除或重写，其余多数被移动了位置，内容相似度只有 50–80%。
- **策略：不做 `git merge`。**
  1. 从 `origin/main` 新建集成分支 `port/main-g1`。
  2. 按模块把 develop 的改动重新实现一遍。
  3. 最后在集成分支上执行 `git merge -s ours develop` 记录 develop 的历史，再让 develop 快进到集成分支（§2）。
- **阶段**
  - P0：让基线在 Linux 上能跑。
  - P1：打地基，让 G1 能加载、旧数据能迁移。
  - P2：四条并行 lane（tracker env / 学习栈 / MDM 与流水线 / 数据工具）。
  - P3：端到端验证。
  - P4：文档和收尾。
  - P5：CVAE，另开分支，放在最后。
- **53 个提交的去向**（逐条见附录 E）
  - 需要移植：37 个，其中大约三分之一需要改造或重新设计。
  - upstream 已经有了：2 个。
  - 已经过时：6 个。
  - revert 对（净改动为零）：6 个。
  - 文档：2 个。
- **关键发现**
  - **main 在 Linux 上不能直接运行。** 包目录叫 `PARC/`，代码却 `import parc` ✔；`DATA_DIR` 写死成 macOS 路径，而且每次加载配置都会断言这个路径存在 ✔。
  - **main 加载不了 G1 的 MJCF。** 默认 joint 和 geom 的解析会崩溃；即使修好，也会把弧度制的关节限位当成角度，结果小 57 倍。
  - **develop 生成的 `.pkl` 在 main 上都不能直接用。** 动作、地形、MDM checkpoint 都是如此。main 没有任何 unpickle 时的模块重映射 ✔。
  - **upstream 有 30 多个 bug 会挡住 G1 流水线。** 例如：stage 2 的关键字参数写错 ✔；`save_ms_file` 的默认值写成了类型对象 ✔；contact 缓冲写死为 15 个 body；plain PPO 的 logger 签名不匹配 ✔。完整列表见附录 C。
  - **develop 自己也有 bug。**
    - predictor 的 `enabled` 开关从来没生效过 ✔：它一直在训练。
    - CLAUDE.md 里的 obs 维度表是错的 ✔：实际总维度是 2363，`tar_obs` 是 1404。
    - CVAE 分支的教师网络从来没被加载过 ✔。
    - 完整列表见附录 D。
- **需要你拍板的决策**在 §8。其中 D1–D12 必须在 P2 开始前定下来。

---

## 1. 两边的差异（迁移时必须遵守的新约定）

| 方面 | develop | main |
|---|---|---|
| 代码布局 | 根目录下的平铺模块 `anim/`、`envs/`、`learning/`、`diffusion/`、`util/` | 包 `parc/`，含 `anim/`、`motion_tracker/{envs,learning}`、`motion_generator/`、`motion_synthesis/`、`motionscope/`、`util/`（完整映射见附录 B） |
| 入口 | 根目录的 `run.py`、`parc_*.py`、`motion_forge.py` | `scripts/run_tracker.py`、`scripts/parc_*.py`、`scripts/run_motionscope.py`；推荐用 `python -m scripts.X` 或 `pip install -e .` |
| 配置 | `PARC/*.yaml`，用 `../Data`、`../tests` 这样的相对路径 | `data/configs/`，路径写作 `$DATA_DIR/...`，由 `parc/util/path_loader.py` 从 `data/configs/user_config.yaml` 读取 |
| 动作文件格式 | legacy pickle：`frames`（dof）加上 pickle 进去的 `util.terrain_util.SubTerrain` 对象 | MS 容器 `{motion_data, terrain_data, misc_data}`（`parc/util/file_io.py`），存四元数 `joint_rot [N, bodies-1, 4]` |
| MDM 存档 | 整个对象 pickle 成 `.pkl` | `save_checkpoint` 存成 `.ckpt` 字典 |
| 日志 | tensorboardX 或 wandb | 只有 wandb（`use_wandb: false` 时为离线模式），`--logger` 参数被忽略 |
| 已删除的功能 | — | MGDM 子环境、`target_xy` 和 `replan_t` 观测、多进程（`num_workers`）、`terrain_perception/`、`tb_logger`、`dm_vit`、`dm_cnn_mlp` |
| 依赖 | Python 3.8.19，torch 1.13.1（cu117） | Python 3.8.20，torch 2.2.0（应选 cu118 版），新增 scipy、imageio[ffmpeg]、py-cpuinfo，去掉 tensorboardX 和 opencv |

---

## 2. 分支与合并模型

```
origin/main ──► develop_migration (集成分支，worktree ../PARC-migration)
                 ├─ migration/p0-baseline   ┐
                 ├─ migration/p1a-charmodel │ 第 1 波（并行）
                 ├─ migration/p1b-dataio    │
                 ├─ migration/lane-b-learning┘
                 ├─ migration/lane-a-env     ┐
                 ├─ migration/lane-c-pipeline│ 第 2 波（并行，等第 1 波合入）
                 └─ migration/lane-d-datatools┘
develop（冻结，作为参考；最后一步之前不动）
```

- **工作目录。** 每个分支一个 worktree，放在仓库的同级目录，命名为 `../PARC-mig-<部分>`。
  - 每个 worktree 用 `uv sync` 建自己的 `.venv`，再写一行 `isaacgym.pth`。uv 的缓存会做硬链接，所以占不了多少空间。
- **合并方式。** 各部分完成后，由协调者审查，再用 `--no-ff` 合回 `develop_migration`。
- **develop 保留为参考环境。** develop 工作区的 `.venv` 是 torch 1.13 的环境，专门用来做 A/B 对比和读取旧数据，得到参考值。
- **提交规范。**
  - 一个任务对应一个提交。
  - 提交信息格式：`Port <功能> from develop (<源提交 hash>)`。
  - 如果修的是 upstream bug，写 `Fix <bug> (upstream)`。
- **收尾合并**（P4）。不需要 force push：
  ```bash
  git tag archive/develop-pre-port develop
  git switch develop_migration && git merge -s ours develop -m "Record develop history; changes re-implemented on upstream main"
  git switch develop && git merge --ff-only develop_migration
  ```
  这一步会改动 develop，**执行前必须得到你的确认**。
  之后要同步 upstream，直接 `git merge origin/main` 就行。
  注意：`-s ours` 会丢弃 develop 中所有没被移植的内容。附录 E 已经逐条确认过这些内容。

---

## 3. 阶段、任务与依赖

```
P0 基线 ─► P1a 角色模型 ┐
          P1b 数据 IO   ┴─► P1c G1 数据迁移(需数据机) ─┐
                         └─► P2: A env ∥ B 学习 ∥ C 流水线 ∥ D 数据工具 ─► P3 E2E ─► P4 收尾 ─► P5 CVAE
```

尺寸标记：S 是小改动，M 是中等，L 是大。🖥 表示需要 GPU 加 IsaacGym，📦 表示需要 G1 数据集。

### P0 基线可运行（1 个 agent，串行）

| ID | 任务 | 尺寸 |
|---|---|---|
| P0.1 | 创建 worktree 和集成分支：`git worktree add -b port/main-g1 ../PARC-port origin/main` | S |
| P0.2 | 修目录名：`git mv PARC parc` ✔。补上 `parc/motion_generator/utils/__init__.py`，否则非 editable 安装会漏掉这个包。`.gitignore` 里把 `*.egg_info` 改成 `*.egg-info/`，再加 `build/` 和 `.venv/` | S |
| P0.3 | 把 develop 上的 uv 方案搬到 main（细节见下方"P0.3 要点"） | M |
| P0.4 | 处理 DATA_DIR：在 `path_loader` 里加 `PARC_DATA_DIR` 环境变量覆盖，没设置时回退到 `user_config.yaml`。原因是 `user_config.yaml` 写的是 `/Users/mxu/Projects/`，且每次加载都断言它存在 ✔（决策 D5） | S |
| P0.5 | 修 29 处过期路径（`parc/data/assets/...`、`data/terrains/...`、`tools/...`、parc_1 的默认配置名），以及挡住基线的小 bug：logger 签名不匹配 ✔、已失效的 `--mode test2`、stage 脚本遇到多余参数会悄悄回退到默认配置 | S |
| P0.6 | 静态检查（具体见 §7） | S |
| P0.7 | 🖥 人形基线冒烟测试 S0–S8：import 扫描、加载 civilization 示例、tracker 训练、测试、record、MDM 两个 epoch。对应门禁 **G0** | M |

**P0.3 要点**（来自 develop 上已经验证过的 uv 方案，合入提交为 `48be988`）

- **保持不变**
  - Python 3.8.20。
  - 只在 Linux x86_64 上解析依赖。
  - IsaacGym 用 `.venv/lib/python3.8/site-packages/isaacgym.pth` 链接进来：它不进 lock，`uv sync` 之后也还在。
  - 不需要设置 `LD_LIBRARY_PATH`。
  - 依赖里显式带上 `setuptools`、`ninja`、`numpy<2`，以及 IsaacGym 自己的运行时依赖。
- **需要改的**
  - **打包方式。** 把 `setup.py` 的元数据并进 pyproject 的 `[project]`（包名同为 `parc`）。去掉 `package = false`，保留 setuptools 作为 `[build-system]`，把入口 `run-motionscope` 移到 `[project.scripts]`。一旦设置了 `[project].dependencies`，setuptools 就会忽略 `install_requires`。
  - **torch 版本。** 升到 torch 2.2.0 和 torchvision 0.17.0。PyPI 上的 2.2.0 是 cu121 版本，本机驱动可以用；如果一定要 cu118，就加一个显式的 PyTorch index 并把这两个包固定到它上面。
  - **依赖增减。** 加 `imageio[ffmpeg]`。去掉 `tensorboardX`（main 已删 `tb_logger`）和 `opencv-python`。develop 上把 opencv 钉在 `<4.10`，是为了避开 `numpy._core` 别名导致的段错误；main 不再需要 opencv，这个钉版也就不需要了。
  - **.gitignore。** 改成 `*.egg-info/`。
- **已知风险**
  - IsaacGym 的 `isaacgym/torch_utils.py` 用了 `np.float`，numpy 1.24 已经删掉了它。main 和 develop 都没有导入 `torch_utils`，所以目前不受影响。
  - 不要把 CUDA 11 或 cuDNN 8 的目录放进 `LD_LIBRARY_PATH`。

### P1 地基（P1a 和 P1b 各一个 agent 并行；P1c 在数据所在的机器上做）

**P1a 角色模型：让 G1 能加载**

| ID | 任务 | 来源 | 尺寸 |
|---|---|---|---|
| P1a.1 | 加入 G1 资产 `data/assets/unitree_g1/**`（69 个文件，52 MB），用 `git checkout develop -- data/assets/unitree_g1` | 46d8913, 74990f2 | S |
| P1a.2 | 修改 `kin_char_model`：<br>• 解析 `<compiler angle>`，设置 `self._angle_scale`<br>• 默认 joint 类型回退到 hinge<br>• `get_default_geom_type` 遇到 None 时不崩溃<br>• 修好 assert 的格式化字符串<br>• 把名字别名集中到 `BODY_NAME_ALIASES`（脚、手，头可选），新增 `resolve_body_name()` 和 `has_body()`，`get_body_id`、`get_joint_id` 都走别名 | 2a2d597, 447e1b1, e70ea0f | S–M |
| P1a.3 | 修 `torch_util.exp_map_to_axis_angle` 的 NaN：保持原来的运算顺序，只对原始范数做截断（develop 的写法有缺陷，见 DB5） | c5dded1 | S |
| P1a.4 | `geom_util.get_minimal_char_point_samples`：支持 MESH、CYLINDER 和没有 geom 的 body（G1 有 6 个这样的 body） | 71933aa | S |
| P1a.5 | CPU 单测：<br>• G1 得到 (36, 36, 29)<br>• `left_knee` 限位约为 [-0.0873, 2.8798]<br>• 别名 `left_foot` 解析为 `left_ankle_roll_link`<br>• 人形模型输出与修改前的快照逐位一致 | — | S |

**P1b 数据 IO 兼容**

| ID | 任务 | 来源 | 尺寸 |
|---|---|---|---|
| P1b.1 | 新建 `parc/util/pickle_compat.py`，提供 `CompatUnpickler`，负责：<br>• 模块重映射：`util.`→`parc.util.`，`anim.`→`parc.anim.`，`tools.motion_opt.`→`parc.motion_synthesis.motion_opt.`，`zmotion_editing_tools.`→`parc.util.`<br>• `numpy._core` 的 shim：只在 numpy < 2 时启用<br>• 把 CUDA 存储映射到 CPU | ba6b896, c67e2a8 | S |
| P1b.2 | 在 `file_io.load_ms_file`（4 处加载点）和 `motion_edit_lib.load_motion_file` 里接入兼容层；读到 legacy 格式时给出明确的报错 | — | S |
| P1b.3 | 修 `scripts/convert_legacy_motions.py`：加 remap，补 fps 和 loop_mode 的默认值，加 `terrain.update_old()`，支持只有地形的文件。删除 `parc/motion_synthesis/` 下的过期副本 | — | M |
| P1b.4 | 修 upstream 的数据 bug：<br>• `BodyConstraint` 这个 dataclass 没有字段注解，导致约束被悄悄丢掉；`torchify_body_constraints` 的两个分支都要修<br>• `save_ms_file` 的默认值从类型对象改成 None ✔<br>• motion_lib 在零 contact 时的回退应该用 `get_num_contact_bodies` | — | S |
| P1b.5 | CPU 单测：numpy 2 生成的 pickle、CUDA 张量的 remap、BodyConstraint 存取往返 | — | S |

**P1c G1 数据迁移（📦，依赖 P1a 和 P1b）**

1. 转换数据，不要原地转换：
   ```bash
   python -m scripts.convert_legacy_motions --inputs $DATA_DIR/Data/initial_retargeted_g1_processed ... --outputs <新目录> --char-file data/assets/unitree_g1/g1_mocap_29dof.xml
   ```
2. 用 `create_dataset_yaml_from_config` 重建数据集 yaml，设 `compute_preprocessing_data: true`。注意它会原地重写文件。
3. 做往返校验，即门禁 **G1**：
   - 在 develop 的 uv 环境里用 `MotionLib` 读旧文件，记下参考值。
   - 在 port 上用 `MotionLib.from_file` 读新文件。
   - 帧数和 fps 必须一致；root_pos 误差 < 1e-6；hinge DOF 误差 < 1e-4 rad；contacts、hf、`hf_mask_inds` 完全相等。
4. 删除旧的 `terrain.pkl` 和 `sampler.pkl` 缓存。

### P2 四条并行 lane（4 个 agent，开始前冻结 §6 的契约）

**Lane A：tracker env**
来源：d5dad54、e7d8c2b、e70ea0f（env 部分）、f9a462b、1ea5a9b。upstream 的 `ig_char_env.py` 基本没改（R098），develop 的 diff 只需按行号偏移套用。

| ID | 任务 | 尺寸 |
|---|---|---|
| A1 | `ig_char_env`：<br>• 新增 `_get_kin_dof_limits()`，写入 `dof_prop` 的 lower/upper；角色和 `_build_ref_character` 都要写（develop 漏了后者）<br>• 动作边界的数据来源改为 kin limits，球关节保留 upstream 逻辑（D7）<br>• 从 `env.pd_params` 读 PD 增益（大小写不敏感的子串匹配，先匹配到的生效）<br>• 用约 20 行的 `_print_dof_props` 替代 develop 那 113 行的 verify<br>• `_build_body_ids_tensor` 走 `resolve_body_name`<br>• `_check_char_model` 加长度断言 | M |
| A2 | debug_visuals：<br>• `env_builder.build_env(..., debug_visuals=False)`，并在 `run_tracker.py` 加 `--debug_visuals`<br>• 射线绘制改成遍历所有 DM env，每帧只做一次 `.cpu()`；`_draw_point` 现在每画一个点就同步一次 GPU | S |
| A3 | 修 upstream 的 env bug：<br>• `range(15)` 的接触着色改成用 contact body ids<br>• `_motion_lib` 已改名为 `_mlib`，但 `ig_parkour_env:362` 和 `dm_motion_recorder:100,137` 还有残留引用，会让 record 模式崩溃<br>• 断言 `fraction_dm_envs == 1.0`（MGDM 已被删除，小于 1 会出错） | S |
| A4 | 配置：<br>• `data/configs/tracker_config/dm_env_g1.yaml`：以 main 的默认配置为模板，修改 char_file、`contact_weights`（36 项）、`init_pose`、`joint_err_w`、`pose_termination_dist`（各 35 项）、`key_bodies`，加入 `pd_params`，删掉已经没人读的键<br>• `data/configs/parc_3_tracker_g1.yaml` | M |
| A5 | 🖥 验收：<br>• 质量约 34.38 kg（如果是约 39.4 kg，说明无质量的 body 被赋了默认质量）<br>• 29 行 DOF 打印与配置一致<br>• `OBS SHAPES` 的输出与附录 A 一致（总维度 2363）<br>• 随机策略跑 test 模式不出现 NaN | M |

不移植：develop 的 113 行 verify 函数、`_apply_action` 里注释掉的调试代码、`dm_env_default.yaml` 的纯外观改动。

**Lane B：学习栈（encoder 和 predictor）**
来源：34d6f97、c107f83、e04ee11、809c625、418cf16、be3d064、4fc8662、57e590d、b956057、4cb655c、e786aaa。
设计原则：用新文件里的子类实现，不改 upstream 的类。保持 develop 的属性名，这样 develop 的 checkpoint 仍能加载。

| ID | 任务 | 尺寸 |
|---|---|---|
| B1 | 在原文件里做的小修：<br>• exp buffer 的 contact 维度改为按 body 数（upstream 写死 15，G1 会崩）<br>• `Normalizer` 对 `min_std` 做截断<br>• 动作 normalizer 的 `min_std`（可选）<br>• 统一用 `_check_nan`，受 `NANCHECK` 控制，出错时以**非零**状态退出 | S–M |
| B2 | 新文件：`learning/modules/{__init__,mlp_encoder}.py`（原样照搬）。`learning/obs_encoder.py` 不再持有 env，并增加检查：各切片总和必须等于 obs 维度 | M |
| B3 | 新文件 `learning/dm_ppo_fp_model.py`，类 `FuturePoseDMPPOModel(DMPPOModel)`：先建 encoder；predictor 的输入是 encoder 没用到的 key；提供 `compute_predictor_loss()` 和 test 模式下的覆盖路径 | L |
| B4 | 新文件 `learning/dm_ppo_fp_agent.py`：<br>• 从 `model.future_pose_predictor` 读 `enabled` 和 `learning_rate`，修掉 DB1 和 DB2 两个 develop bug，读学习率时要转 `float()`<br>• predictor 用独立的 Optimizer，并从主优化器里排除它的参数<br>• 每步之后清掉 predictor 的梯度 | M |
| B5 | `agent_builder` 的分派：`DM_PPO` 且 `model.obs_encoders` 非空时用子类。`run_tracker.py` 加 `--test_predictor`，train 模式下禁止使用 | S |
| B6 | `data/configs/tracker_config/dm_agent_g1.yaml`：学习率写成 `1.0e-4`，否则 yaml 会解析成字符串；`enabled` 的取值见决策 D1 | S |
| B7 | `tests/test_future_pose.py`，用 FakeEnv 在 CPU 上跑：<br>• 形状 2363→775，透传切片正确，latent 是单位范数<br>• predictor 在固定 batch 上 300 步后 loss 下降到 1/5 以下，且其他参数不变<br>• test 模式只改变 actor，不改变 critic<br>• `enabled: false` 时不训练 predictor、也不把它加进优化器<br>• 默认配置的参数量等于 upstream 的 | M |

**Lane C：MDM、流水线、motionscope**
来源：d8f2c17、932d1f4、98b873e、f4bb734、10c074c、4cbe311（可选）、342741a、17969d5、39bc984、bacd697，以及 74990f2 的配置部分。

| ID | 任务 | 尺寸 |
|---|---|---|
| C1 | MDM：<br>• `compute_point_hf_sdf` 改为逐 body 计算<br>• parc_1 在 sampler 的设备与配置不符时重建 sampler（develop 的 `update_old_sampler(config)` 有签名 bug，不要移植）<br>• `assemble_mdm_features` 加设备保护 | S |
| C2 | parc_2：<br>• 用 `device` 配置键替代写死的 `cuda:0`<br>• 把 `src_frames=` 改成 `src_motion_frames=`，共 3 处 ✔<br>• 修 CPU/GPU 设备混用<br>• `:377` 处调用 `save_ms_file` 时补上缺的参数 ✔<br>• 是否使用 EMA 权重见 D8 | M |
| C3 | G1 配置变体：<br>• `parc_0_setup_iter_config_g1.yaml`、`parc_1_train_gen_g1.yaml`（超参见 D3）、`parc_2_kin_gen_g1.yaml`、`create_dataset_config_g1.yaml`（`min_num_frames: 16`，因为 sampler 会对恰好 15 帧的动作断言失败）<br>• parc_0 可以选择用一个 `char_file` 统一下发到各阶段 | S–M |
| C4 | 数据集和 stage 4：<br>• create_dataset 跳过 `log/` 目录<br>• 空类别的处理策略见 D10<br>• 断言信息里 str 加 float 会抛 TypeError，要修<br>• `create_dataset_yaml.py` 加 `--config`，现在它写死了人形和固定路径，而且一导入就运行<br>• parc_4 的 NameError，以及先写文件后建目录的顺序问题 | S |
| C5 | motion_opt：名字查找走集中的别名表。G1 的 body 约束问题见 D16：脚和手的第一个 geom 是 mesh，所以约束被跳过，develop 上也是这样 | S（如果要做 G1 专用分支则是 M） |
| C6 | motion_tests：`terrains` 支持单个字符串；`get_mlib_format` 这个函数不存在，要改调用；路径改用 `$DATA_DIR` | S |
| C7 | motionscope：<br>• polyscope_util 改用 `.cpu().numpy()`<br>• `_resolve_char_path` 加 g1<br>• 手部名字走别名表<br>• 修 GUI 里的 `get_mlib_format` 调用 | S |
| C8 | ~~把 develop 的 MDM `.pkl` 转成 `.ckpt` 的脚本~~ **已取消**：D2 决定不复用 MDM checkpoint | — |

**Lane D：数据工具**
来源：ca998c8、2f2c6cb、eb0dd19、dbdfa93，以及 74990f2 的工具部分。

| ID | 任务 | 尺寸 |
|---|---|---|
| D1 | 把以下工具搬到 `scripts/tools/`：<br>• `print_model_info`：修默认路径<br>• `pkl_preview` 及其示例：能识别 MS 容器<br>• `pkl_extract_frames_fps`：增加 MS 输入路径<br>• `extract_motion.sh`：修路径<br>• `view_terrain`：同时支持 MS 和 legacy<br>• `obj_to_heightfield`：导入改成 `parc.util.terrain_util`，并修网格间距的 bug（DB7） | M |
| D2 | `scripts/robot_motion_preprocess.py`（GMR 到 PARC 的桥，L）：<br>• 用 CompatUnpickler 读 GMR 输出（numpy 2）<br>• 源动作支持 MS 和 legacy 两种格式<br>• 用 `--source_char_file` 按名字映射 contact，替代写死的列号<br>• 几何 contact 的回退不再使用 mesh 的 dims<br>• 脚底偏移由脚部的球体推算，G1 为 0.035，develop 写的 0.05 会让脚悬空约 1.5 cm<br>• `compute_hf_extra_vals` 用新签名<br>• 输出 MS 格式 | L |
| D3 | 测试：<br>• 合成一个 0.5 m 的 OBJ 盒子，检查 hf 的值和边界<br>• MS 文件预览<br>• 📦 同一份 GMR 输出，分别用 develop 版和 port 版处理，contacts 和 `hf_mask_inds` 必须一致 | M |

### P3 集成与端到端验证（1 个 agent，加上 🖥📦 的人工或 GPU 环节）

- **合并顺序**：A、B、C、D。各 lane 的文件归属不重叠。唯一共享的文件是 `scripts/run_tracker.py`：A 加的 `--debug_visuals` 在 `build_env` 附近，B 加的 `--test_predictor` 在 `agent.load` 之后，两处不冲突。每次合并后都跑一遍静态检查和 CPU 单测。
- **门禁 G3（🖥📦）**：
  1. **G1 tracker 的 A/B 对比**（develop 对 port）：用同一套配置，`enabled: true`，3 个种子，2048 个 env，300–500 次迭代。对比 `Predictor_Loss`（约 0.06 起步，应下降）、Train/Test Return、Episode Length、Critic_Loss、Clip_Frac、body pos tracking error。要求两边在种子波动范围内重叠，且没有 NaN 退出。
  2. **旧 checkpoint 兼容**：develop 第 C 代 checkpoint（附录 F）在 port 上以 test 模式运行，分别带和不带 `--test_predictor`，指标与 develop 上的结果比较。
  3. **G1 流水线冒烟**：stage 0→4 按极小规模跑一遍，参数见 §7。
  4. **record 冒烟**：`--mode record --test_predictor True`。

### P4 文档与收尾（1 个 agent）

- 按新结构重写 CLAUDE.md：用附录 A 的 obs 表；写 uv 命令、`python -m scripts.X`、`$DATA_DIR`；去掉 MGDM、`num_workers`、tensorboard。
- `test_tracker_args.txt` 改成用 `$DATA_DIR` 的模板，或者删掉。
- 按 §2 的方式合入 develop。是否 push 由你决定。
- 可选：向 mshoe/PARC 提 PR，内容包括目录大小写、路径修复和通用 bug 修复，以减少将来同步时的冲突。

### P5 CVAE（另开 `feature/cvae_latent`，在 P4 之后）

- **先做机械移植。** 把 `6213076` 和 `8da78fc` 合并成一个提交，把 3 个文件放到新位置：
  - `parc/motion_tracker/learning/modules/cvae_skill.py`
  - `scripts/cvae_training.py`
  - `data/configs/cvae/cvae_g1.yaml`
- **再修 bug（DB10）。**
  - **教师网络的加载。** 改用 `agent_builder.build_agent` 加 `agent.load`，严格加载；同时补上 obs 归一化和动作反归一化。
  - **重置逻辑。** 只重置已经结束的 env。
  - **DAgger。** 修标签错位的问题；修 buffer 容量，确保能在学生状态上训练。
- **待决问题。** encoder 是否应该把 action 作为输入，需要单独决定。

---

## 4. 不移植的内容（`-s ours` 会把它们丢掉，这是有意的）

- 2 个已经在 upstream 的提交：`d381000`、`7f8651a`。
- 6 个过时的提交：
  - `e3eca62`：upstream 已删除 MDM 的 foot-contact loss。
  - `d5f2f4f`：优化器已改用 MotionFrames API。
  - `f12ad04`、`ab4ee1c`、`2d488b6`：`mdm_tester.py` 在 upstream 已删除，也没有替代（是否要做替代见 D17）。
  - `4cbe311`：parc_3 已经自己写配置。
- 3 对 revert（共 6 个提交）。
- develop 里 MGDM 相关的代码路径、多进程、tensorboard、`target_xy`。
- `imgui.ini`、`MOTION_FORGE/motion_forge_config.yaml`（含机器相关的路径）、`diffusion/mdm.yaml`、`1618531` 清理掉的 legacy `tar_obs_latent_dim` 支持。

---

## 5. 执行所需的资源和前置条件

| ID | 需要什么 | 用在哪里 | 状态 |
|---|---|---|---|
| R1 | IsaacGym Preview 4 | G0、A5、G3 | ✅ 已在 `../isaacgym`（来自 NVIDIA 官方 CDN，sha256 `c51d77ac…fdb8f`） |
| R2 | G1 数据集 | P1c、G3 | ⏸ 暂时没有。P1c 暂缓；在此之前用合成数据和仓库自带的人形示例测试 |
| R3 | tracker C 代 checkpoint | G3 | ⏸ 还没有。兼容检查暂缓；B7 用 FakeEnv 验证 state_dict 的 key 和形状 |
| R4 | `PARC_DATA_DIR` 的值 | P0.4 | ✅ 仓库的父目录 `/home/ubuntu/projects/parkour_sgs` |

**本机环境的约束（所有 agent 都必须遵守）**

- 不要动本机的 CUDA 环境，因为其他项目也在用这台机器：
  - 不用 sudo，不用 apt，不改驱动和系统 CUDA。
  - 不全局设置 `LD_LIBRARY_PATH`，不改 `~/.bashrc` 这类文件。
  - 只在项目自己的 `.venv` 里装包。
- 所有训练和测试命令都要带 `WANDB_MODE=disabled`，因为这台机器登录着团队的 wandb。
- 共用一块 L40S，要照顾其他人：
  - 仿真前先用 `nvidia-smi` 看一下显存。
  - 冒烟测试的 `num_envs` 不超过 64，时间尽量短。
  - 跑完确认没有残留进程。

---

## 6. 多 agent 编排

**执行波次（实际采用的方案）**

P0.1–P0.3 由协调者在 `develop_migration` 上直接完成。剩下的工作分成两波，每个 agent 一个 worktree，从 `develop_migration` 开分支：

| 波次 | Agent | 分支 | 内容 |
|---|---|---|---|
| 1 | P0 剩余部分 | `migration/p0-baseline` | P0.4 `PARC_DATA_DIR`、P0.5 过期路径和小 bug、P0.6 静态检查、P0.7 人形基线冒烟 |
| 1 | P1a | `migration/p1a-charmodel` | 见 P1a |
| 1 | P1b | `migration/p1b-dataio` | 见 P1b |
| 1 | Lane B | `migration/lane-b-learning` | 见 Lane B。它不依赖 P1，所以提前开始。P0.5 里的 logger 签名修复和 `--mode test2` 也交给它，因为它独占 learning 目录和 `run_tracker.py` |
| 2 | Lane A、C、D | `migration/lane-{a-env,c-pipeline,d-datatools}` | 依赖 P1a 和 P1b，等第 1 波合入后再开始 |
| 之后 | P3 集成、P4 文档 | `develop_migration` | 部分门禁要等 R2 和 R3 |

第 1 波的文件归属：
- P0 不碰以下文件：`parc/motion_tracker/learning/**` 和 `scripts/run_tracker.py`（归 B）；`parc/anim/**`、`parc/util/{torch_util,geom_util}.py`（归 P1a）；`parc/util/{file_io,file_io_helper,motion_edit_lib}.py`、`parc/motion_synthesis/motion_opt/motion_optimization.py` 和两个 `convert_legacy_motions.py`（归 P1b）。
- P1b 拥有 `parc/anim/motion_lib.py`；P1a 只拥有 `parc/anim/kin_char_model.py`。
- 测试统一放在 `tests/` 下，每个 agent 用自己的文件名，避免冲突。

**文件归属（P2，避免冲突）**

| Lane | 独占的文件 |
|---|---|
| A | `parc/motion_tracker/envs/{ig_char_env,ig_env,env_builder}.py`、`envs/ig_parkour/ig_parkour_env.py`、`learning/dm_motion_recorder.py`（只修 `_mlib`）、`data/configs/tracker_config/dm_env_g1.yaml`、`data/configs/parc_3_tracker_g1.yaml` |
| B | `parc/motion_tracker/learning/{base_agent,ppo_agent,dm_ppo_agent,normalizer,agent_builder}.py`；新文件 `learning/{obs_encoder,dm_ppo_fp_model,dm_ppo_fp_agent}.py`、`learning/modules/`、`data/configs/tracker_config/dm_agent_g1.yaml`、`tests/` |
| C | `parc/motion_generator/mdm.py`、`scripts/{parc_0,parc_1,parc_2,parc_4,create_dataset_yaml,run_optimize_motions,run_simple_augment_motions}.py`、`parc/util/create_dataset.py`、`parc/motion_synthesis/**`、`scripts/motion_tests/**`、`parc/motionscope/**`、`data/configs/{parc_0,parc_1,parc_2,create_dataset}*_g1.yaml` |
| D | 新文件 `scripts/robot_motion_preprocess.py`、`scripts/tools/**` |
| 共享 | `scripts/run_tracker.py`：A 和 B 各改一处，位置不重叠 |

**冻结的契约（P2 开始前由协调者确认，lane 期间不能改）**

- **C-obs**：obs 字典的 key、形状和顺序，以附录 A 为准。A5 门禁会用运行时的 `OBS SHAPES` 打印来确认。
- **C-alias**：`KinCharModel.resolve_body_name(name) -> str`、`has_body(name) -> bool`；`get_body_id` 也会走别名。
- **C-limits**：`Joint.limits` 的单位是弧度；`_lower_dof_limits` 和 `_upper_dof_limits`；KinCharModel 的 DOF 顺序与 IsaacGym 一致（深度优先）。
- **C-geom**：`get_minimal_char_point_samples(cm) -> List[Tensor[k_b>=1, 3]]`，长度等于 body 数。
- **C-io**：`pickle_compat.load/loads(..., map_location="cpu")`；MS 格式保持 upstream 不变。
- **C-cfg/cli**：G1 配置的文件名按上面的表；`build_env(env_file, num_envs, device, visualize, debug_visuals=False)`；`--debug_visuals` 归 A，`--test_predictor` 归 B。

**每个 lane agent 的完成标准**

1. 只改自己归属的文件。需要改其他文件时，报告给协调者，不要自己动手。
2. 每个任务一个提交，遵守 §2 的提交规范。不要 push。
3. 静态检查和 CPU 单测全部通过。需要 🖥📦 的检查列为待验证项。
4. 最终报告包含：任务清单的完成状态、证据（命令和输出片段）、待 GPU 验证的列表、有疑问的地方。

**协调者的职责**

- 确认契约。
- 审查每个 lane 的 diff。
- 按顺序合并，合并后跑检查。
- 更新本文档的状态。
- 把决策问题汇总给你。

---

## 7. 验证矩阵

**静态检查（本机可以跑，用 main 的 uv 环境，不需要 IsaacGym）**

- `python -m compileall -q parc scripts`：使用 3.8 解释器。
- import 扫描：先导入 IsaacGym 的 stub 或者跳过依赖 IsaacGym 的模块；跳过导入时就会执行操作的脚本，例如 `create_dataset_yaml`、motionscope。
- 检查没有残留的旧导入：`git grep -nE "^(import|from) (util|anim|learning|envs|diffusion|tools|zmotion_editing_tools)\b" -- parc scripts` 应该没有输出。
- 检查没有残留的机器相关路径：`git grep -n "/home/phi\|/Users/mxu" -- parc scripts data/configs` 应该没有输出。

**CPU 单测**：P1a.5、P1b.5、B7、D3（合成数据部分）。

**门禁**

| 门禁 | 内容 | 环境 |
|---|---|---|
| G0 | 人形基线 S0–S8 | 🖥 |
| G1 | G1 能加载（36/36/29，限位为弧度），人形保持不变；G1 数据往返一致 | CPU + 📦 |
| G2 | A5 env 验收；B7 单测；C 的每个 stage 极小规模冒烟；D3 | 🖥📦 |
| G3 | §3 P3 中的 4 项 E2E 检查 | 🖥📦 |

**极小规模冒烟参数（Lane C 和 G3 用）**

- Stage 1：`epochs: 2`，`iters_per_epoch: 2`，`epochs_per_checkpoint: 1`，`batch_size: 4`，`use_wandb: false`。
- Stage 2：`num_new_motions: 1`，`mdm_batch_size: 4`，`opt.num_iters: 10`，各 loss 阈值设为 1e9，因为未训练的 MDM 达不到正常阈值。
- Stage 3：`num_envs: 4`，`max_samples` 约 2e4。
- Stage 4：使用 p3 产出的 `model.pt`，期望得到按类别分箱的 `recorded_motions`。

---

## 8. 需要你决策的问题

**已经定下来的（2026-10-14 与你逐项确认）**

| ID | 问题 | 结论 |
|---|---|---|
| D1 | G1 模板里 predictor 的 `enabled` | **`true`**，与 develop 的实际行为一致 |
| D2 | 哪些 checkpoint 必须能加载 | **只要求 tracker C 代**（保持属性名）。MDM 不转换，C8 取消 |
| D3 | G1 的 MDM 超参 | **沿用 develop 的**：1001 步、1024 维、hf 碰撞和 target loss 都开。所以 d8f2c17 和 71933aa 是必须的 |
| D4 | 旧数据在哪边转换 | **在 main 上用 CompatUnpickler 转换** |
| D5 | DATA_DIR 怎么配置 | **用 `PARC_DATA_DIR` 环境变量**，没设置时回退到 `user_config.yaml` |
| D6 | 名字别名放哪里 | **集中放在 KinCharModel**（`BODY_NAME_ALIASES` 和 `resolve_body_name`） |
| D7 | 球关节动作边界 | **保持 upstream 的** |
| D8 | stage 2 是否用 EMA 权重 | **用 EMA** |
| D9 | kin_gen 起始朝向 | **`auto`** |
| D10 | 数据集策略 | **develop 风格**：空类别警告后跳过；`cut_some_classes_in_half: true`，`max_terrain_dim: 45` |
| D11 | 是否给 upstream 提 PR | **先只修在移植分支上**，PR 以后再说 |
| D12 | G1 的 contact body 数 | **保持 36 个** |

**可以晚点定**

| ID | 问题 |
|---|---|
| D13 | 是否通过配置施加 G1 的力矩限制（`actuatorfrcrange`）？ |
| D14 | `key_bodies` 用 `head_link`（几乎在躯干原点）还是 `head_mocap`？G1 的终止条件非常宽松，是有意的吗？ |
| D15 | hf 调试绘制：画所有 env，还是设一个上限？ |
| D16 | motion_opt 要不要为 G1 加专门分支？目前 G1 的 body 约束完全无效，脚的偏移取的是踝关节原点 |
| D17 | 是否需要 `mdm_tester` 的替代（可视化 sampler batch 和预测结果）？ |
| D18 | parc_2 的 FILE 模式（.obj 生成地形）要用于 G1 吗？如果要，需要修 U4 |
| D19 | 52 MB 的 G1 资产全部保留，还是只留 `g1_mocap_29dof.xml` 用到的网格？要不要用 Git LFS？ |
| D20 | 工具里的中文注释要不要翻译成英文？ |
| D21 | 809c625 里的动作调试打印要保留吗？ |
| D22 | `test_predictor` 要不要也下发到 parc_4 的 record 配置？ |

---

## 9. 风险

| 风险 | 应对 |
|---|---|
| IsaacGym 与 torch 2.2 的组合：gymtorch 需要针对已安装的 torch 用 JIT 重新编译，要求有 ninja 和支持 C++17 的编译器，旧的 `~/.cache/torch_extensions/py38_cu*/gymtorch` 必须清掉 | P0.7 第一项检查就做这个；uv 环境里带上 ninja |
| G1 数据转换的保真度 | P1c 的往返校验门禁 |
| upstream 对 MDM 的改动（删除了一些 loss，默认超参也大改）会影响 G1 的动作生成质量 | D3 先保持 develop 的超参；G3 中做对比 |
| 修好 `enabled` 开关后，行为会变化 | D1 设为 `true` |
| 日志改成只有 wandb，tensorboard 的工作流会失效 | 在文档里写清楚；离线模式可用 |
| `torch.load` 没有设置 `weights_only`，torch ≥ 2.6 会出问题 | 备注；锁定 torch 2.2 |
| `-s ours` 会丢掉没有移植的内容 | 附录 E 已逐条确认；保留 archive tag |

---

## 附录 A：G1 观测契约（main 与 develop 的 G1 配置一致）

数据来源：
- 代码路径：`ig_char_env` 中 `joint_rot = dof_to_rot(dof_pos)`，以及 `compute_char_obs`；`mgdm_dm_util.compute_tar_obs` 拼接的是 root_pos、root_rot、joint_rot 和 key_pos。
- G1 的 MJCF 解析结果：36 个 body，即根 + 29 个 hinge + 6 个固定关节 ✔。
- 在 develop 的 uv 环境里实测：G1 的 `dof_to_rot` 返回 **35 个关节**，按这个算出的维度就是下表的数字 ✔。

A5 门禁还要再用运行时的 `OBS SHAPES` 打印确认一次。

注意：学习栈分析报告和 uv 冒烟测试里出现过 2021/739 和 2024/742 这两组数，都是从旧 CLAUDE.md 推出来的，**是错的**。

| # | key | 形状 | 扁平区间 | 是否归一化 | 条件 | 内容 |
|---|---|---|---|---|---|---|
| 1 | `char_obs` | (266,) | [0, 266) | 是 | 总是 | root_rot 6 + root_vel 3 + root_ang_vel 3 + joint_rot 35×6=210（**包含 6 个固定关节**） + dof_vel 29 + key_pos 5×3=15 |
| 2 | `tar_obs` | (6, 234) | [266, 1670) | 是 | `enable_tar_obs` | 每步：root_pos 3 + root_rot 6 + joint_rot 210 + key_pos 15；6 步的时间为 [1, 2, 3, 10, 20, 30]/30 s |
| 3 | `tar_contacts` | (6, 36) | [1670, 1886) | 否 | `use_contact_info` 且 `enable_tar_obs` | 参考动作的 36 个 contact body |
| 4 | `char_contacts` | (36,) | [1886, 1922) | 否 | `use_contact_info` | 二值，阈值为 \|F\| > 1e-5 |
| 5 | `hf` | (441,) | [1922, 2363) | 否 | 总是 | 7 条射线 × 63 个点；第 i 条射线方向为 −45° + 15°·i；采样距离 −0.10 到 +3.00 m，步长 0.05；数值为相对根部的高度，截断到 ±3 |

**合计 2363**（CLAUDE.md 原来写的 1098、174、"~2024" 都是错的）。

各部分的维度：
- encoder 的输入（`tar_obs` + `tar_contacts`）：1620
- predictor 的输入（`char_obs` + `char_contacts` + `hf`）：743
- actor 和 critic 的输入：775

编码后的布局：

| 区间 | 内容 |
|---|---|
| [0, 266) | `char_obs` |
| [266, 298) | latent |
| [298, 334) | `char_contacts` |
| [334, 775) | `hf` |

## 附录 B：路径映射（develop → main）

| develop | main |
|---|---|
| `anim/` | `parc/anim/` |
| `diffusion/*`（`motion_generator.py` 对应 `base_generator.py`） | `parc/motion_generator/*` |
| `envs/*` | `parc/motion_tracker/envs/*`（删除了 `mgdm_env.py`、`ig_deepmimic_env.py`、`terrain_perception/`） |
| `learning/*` | `parc/motion_tracker/learning/*`（`dm_ppo_agent.py` 重写，`dm_ppo_return_tracker.py` 对应 `return_tracker.py`，新增 `optimizer.py` 和 `dm_motion_recorder.py`） |
| `learning/nets/` | `parc/util/nets/` |
| `util/arg_parser.py` | `parc/motion_tracker/util/arg_parser.py` |
| `util/*` | `parc/util/*`（删除了 `mp_util`、`tb_logger`、`util.py`、`circular_buffer`） |
| `zmotion_editing_tools/motion_edit_lib.py` | `parc/util/motion_edit_lib.py` |
| `MOTION_FORGE/`、`motion_forge.py` | `parc/motionscope/`、`scripts/run_motionscope.py` |
| `run.py` | `scripts/run_tracker.py` |
| `parc_{0..4}_*.py` | `scripts/parc_{0..4}_*.py` |
| `parc_5_create_dataset.py` | `scripts/create_dataset_yaml.py` |
| `tools/motion_opt/` | `parc/motion_synthesis/motion_opt/`，以及 `scripts/run_optimize_motions.py`、`run_simple_augment_motions.py` |
| `tools/procgen/` | `parc/motion_synthesis/procgen/` |
| `tools/motion_tests/` | `scripts/motion_tests/` |
| `PARC/*.yaml`、`PARC/tracker_config/` | `data/configs/parc_*_default.yaml`、`data/configs/tracker_config/` |
| `data/terrains/` | `data/motion_terrains/` |

## 附录 C：upstream bug 清单（会挡住或影响 G1 移植）

| ID | Bug | 核实 | 任务 |
|---|---|---|---|
| UB1 | 目录叫 `PARC/` 但代码 `import parc`，Linux 上导入失败 | ✔ | P0.2 |
| UB2 | `motion_generator/utils/` 缺 `__init__.py`；`.gitignore` 里 egg-info 拼写错误 | | P0.2 |
| UB3 | `user_config.yaml` 里的 DATA_DIR 是 macOS 路径，且每次加载都断言它存在 | ✔ | P0.4 |
| UB4 | 29 处过期路径，包括 civilization 示例、parc_1 默认配置名、`tools/...` | | P0.5 |
| UB5 | `BaseAgent._build_logger` 的签名不匹配，导致 plain PPO 崩溃 | ✔ | P0.5 |
| UB6 | `--mode test2` 调用的函数已被删除；stage 脚本遇到多余参数会悄悄回退到默认配置 | | P0.5 |
| UB7 | parc_2 传的是 `src_frames=`，函数参数名是 `src_motion_frames`（另外两个脚本也一样），所以 `use_opt` 必然报 TypeError | ✔ | C2 |
| UB8 | parc_2 中 CPU 和 GPU 设备混用 | | C2 |
| UB9 | `save_ms_file` 的默认值写成了类型对象；parc_2:377 的调用缺少必需参数 | ✔ | P1b.4 / C2 |
| UB10 | parc_2 不使用 EMA 权重（行为变化） | | C2 / D8 |
| UB11 | parc_2 的 FILE 模式读取 legacy pickle | | D18 |
| UB12 | sampler 对恰好 seq_len 帧的动作断言失败，而默认 `min_num_frames` 是 15 | | C3 |
| UB13 | `create_dataset_yaml.py` 写死了人形和路径，而且一导入就运行；断言信息里 str 加 float 会抛 TypeError | | C4 |
| UB14 | parc_4 没有 `create_dataset_config` 时报 NameError；先写文件后建目录 | | C4 |
| UB15 | parc_1 需要的 `sampler_stats_filepath` 不在默认 yaml 里 | | C3 |
| UB16 | 调用了不存在的 `get_mlib_format`（mdm_validation_test、path_planning_gui、isaac_gym_gui） | | C6 / C7 |
| UB17 | `BodyConstraint` 是没有字段注解的 dataclass，导致约束被悄悄丢掉；`torchify` 的两个分支都有错 | | P1b.4 |
| UB18 | motion_lib 在零 contact 时的回退用了 `get_num_joints` | | P1b.4 |
| UB19 | 转换器没有模块重映射、没有默认值、不调用 `update_old`、不支持只有地形的文件；还有一份过期副本 | ✔（没有 remap） | P1b.3 |
| UB20 | kin_char_model 不支持弧度制，默认 joint 和 geom 的解析会崩溃 | | P1a.2 |
| UB21 | DMPPOAgent 的 exp buffer 把 contact 写死为 15 | | B1 |
| UB22 | `range(15)` 的接触着色；`_motion_lib` 的残留引用让 record 模式报 AttributeError；`fraction_dm_envs < 1` 时维度不匹配 | | A3 |
| UB23 | `_check_char_model` 用 zip 比较却不检查长度；`_draw_point` 每画一个点同步一次 GPU | | A1 / A2 |
| UB24 | NaN 时用 `exit()` 退出，返回码为 0；agent yaml 里的 `1e-4` 被解析成字符串 | | B1 / B4 |
| UB25 | motionscope 调用了不存在的函数：`build_contact_terrain_heuristic`、`generate_cave`、`body_constraint_to_numpy_recursive` | | C7（可选） |
| UB26 | motion_opt 的 body 约束只处理首个 geom 是 SPHERE 或 BOX 的情况，对 G1 无效 | | D16 |
| UB27 | `torch.load` 没有设置 `weights_only`；terrain "file" 模式重新加载时格式不一致 | | 备注 |

## 附录 D：develop 自身的 bug（移植时修掉，不要照搬）

| ID | Bug | 核实 | 处理 |
|---|---|---|---|
| DB1 | predictor 的 `enabled` 从顶层配置读，而 yaml 把它放在 `model:` 下面，所以开关从来没生效，predictor 一直在训练 | ✔ | B4 |
| DB2 | predictor 的学习率写死为 1e-4，yaml 里的值被忽略 | ✔ | B4 |
| DB3 | CLAUDE.md 的 obs 维度表错误 | ✔ | uv 合入时修；P4 |
| DB4 | parc_1 的 `update_old_sampler(config)` 签名不匹配，是潜在的 TypeError | | 不移植 |
| DB5 | `exp_map_to_axis_angle` 先归一化再截断：当 \|θ\| 在 (π, 2π) 之间时，返回的轴长约为 1e5 | | P1a.3 |
| DB6 | `anim/motion_lib.py` 在全局把 `numpy._core` 指向 `numpy.core`：在 numpy ≥ 2 时会覆盖真正的模块；之后再导入任何基于 NumPy 2 构建的扩展（例如 opencv ≥ 4.10）都会段错误，develop 上因此把 opencv 钉在 `<4.10` | ✔（uv 测试中复现） | P1b.1：只在 unpickler 内部做重映射，不在全局起别名 |
| DB7 | `obj_to_heightfield` 的网格间距不等于 dx，`min_point` 没考虑 border_offset | | D1 |
| DB8 | `robot_motion_preprocess`：脚底偏移 0.05 让脚悬空约 1.5 cm；用 mesh 的 dims 做离地检查，结果永远通过；contact 的列号是写死的 | | D2 |
| DB9 | `extract_motion.sh` 的路径错误 | | D1 |
| DB10 | CVAE：教师网络从未加载 ✔；任何一个 env 结束就重置全部 env ✔；DAgger 标签错位；DAgger 从未在学生状态上训练；没有 obs 和动作的归一化 | 部分 ✔ | P5 |
| DB11 | develop 把球关节的动作边界改成逐轴，改变了人形的动作空间 | | 不移植（D7） |
| DB12 | `test_tracker_args.txt` 写死了 `/home/phi` 的路径 | | P4 |

## 附录 E：53 个提交的去向

| 提交 | 内容 | 去向 | 任务 |
|---|---|---|---|
| d381000 | MDM checkpoint 路径拼接 | upstream 已有 | — |
| 4d0a288, ec8803e | revert 对（parc_3） | 净零 | — |
| 46d8913 | G1 资产 | 移植 | P1a.1 |
| ca998c8 | pkl 预览工具 | 移植并改造（MS 格式） | D1 |
| ba6b896 | `numpy._core` 兼容 | 移植并改造（加条件） | P1b.1 |
| 2f2c6cb | `robot_motion_preprocess` | 重新设计 | D2 |
| e3eca62 | MDM 适配 G1 的脚名 | 过时（upstream 已删 foot-contact loss） | — |
| d8f2c17 | 逐 body 计算 SDF | 移植 | C1 |
| 71933aa | geom 点采样 | 移植 | P1a.4 |
| bacd697 | polyscope 的 GPU 转 numpy | 移植 | C7 |
| 932d1f4 | 在其他 GPU 上恢复训练 | 部分移植（sampler 部分） | C1 |
| 7f8651a | 恢复训练时覆盖 epochs 和 batch_size | upstream 已有 | — |
| eb0dd19 | `view_terrain` | 移植并改造 | D1 |
| 2a2d597 | 脚名别名 | 移植并改为集中的别名表 | P1a.2 |
| 98b873e, f4bb734 | parc_2 的设备 | 部分移植（remap 已过时） | C2 |
| dbdfa93 | `obj_to_heightfield` | 移植并修 bug | D1 |
| d5f2f4f | motion_opt 写死的 dof_size | 过时 | — |
| c5dded1 | exp_map 的 NaN | 移植（修正版） | P1a.3 |
| f12ad04, ab4ee1c, 2d488b6 | `mdm_tester` | 过时（文件已删除） | D17 |
| 10c074c | setup 中下发 char_file | 移植为 G1 配置变体 | C3 |
| 4cbe311 | parc_0 预先写 agent_config | 基本过时（可选） | C3 |
| e04ee11, c107f83 | normalizer 的 `min_std` | 移植 | B1 |
| ec84b91, d2493b5 | revert 对（kin_gen yaml） | 净零 | — |
| 447e1b1 | joint 解析 | 部分移植 | P1a.2 |
| 34d6f97 | contact 缓冲大小 | 移植 | B1 |
| d5dad54 | env 中多个脚和手的映射 | 移植（走别名表） | A1 |
| 1ea5a9b | G1 tracker 配置 | 按新 schema 移植 | A4 / B6 |
| 17969d5 | motion_opt 的 body 名映射 | 移植（走别名表） | C5 |
| 39bc984 | validation test 支持单个地形 | 移植 | C6 |
| e7d8c2b | 从配置读 PD 参数 | 移植 | A1 |
| c67e2a8 | 跨设备序列化 | 大部分 upstream 已有，剩余部分并入 pickle_compat | P1b.1 |
| e70ea0f | 弧度单位 | 移植（kin 部分和 env 部分） | P1a.2 / A1 |
| 418cf16, 4cb655c | obs encoder、多 key 组 | 移植 | B2 / B3 |
| 809c625 | NaN 检查 | 移植（改为非零退出） | B1 |
| 342741a | create_dataset 的除零 | 部分移植 | C4 |
| 74990f2 | G1 默认配置（涉及多个文件） | 拆分：xml→P1a.1，配置→A4、B6、C3，工具→D1，其余过时 | 多个 |
| 305da4c, 0e2f986 | revert 对（MLPEncoder 移动） | 净零 | — |
| be3d064 | MLPEncoder 单独成文件 | 移植 | B2 |
| 4fc8662, b956057 | predictor 训练和 test 模式 | 移植 | B3–B5 |
| 57e590d | predictor 配置 | 移植 | B6 |
| f9a462b | hf 可视化和 `debug_visuals` | 重新设计（只保留 DM） | A2 |
| e786aaa | `enabled` 开关 | 移植并修 bug（DB1） | B4 |
| 53b6a67, 1618531 | CLAUDE.md 和 legacy 清理 | 文档重写；legacy 支持不移植 | P4 |

另外，`origin/fix/ckpt_saving`（16ae722）与 d381000 是同一个补丁（用 `git cherry` 核实过 ✔），不需要单独处理。`origin/test/cvae_latent` 见 P5。

## 附录 F：tracker checkpoint 的代际（决定能否在 port 上加载）

| 代 | 对应的 develop 提交 | encoder 的 key | predictor | 在 port 上加载需要的配置 |
|---|---|---|---|---|
| A | 418cf16 之前 | 无 | 无 | 不配 `obs_encoders`，即 upstream 原版 agent |
| B1 | 418cf16 到 be3d064 | `_encoders.tar_obs` | 无 | `obs_encoders: {tar_obs: ...}` |
| B2 | 4fc8662 到 f9a462b | `_encoders.tar_obs` | 有 | B1 的配置，加上 `future_pose_predictor.target_encoder: tar_obs` |
| C | 4cb655c 到 10ff9af | `_encoders.future_pose` | 有 | 直接用 `dm_agent_g1.yaml` |

前提是每个 key 的 obs 维度与附录 A 一致。优化器状态从来没有保存过，所以不需要转换。`test_tracker_args.txt` 里的 `model_0000007800.pt` 是 2025-12-19 的，属于 A 代或 B1 代。
