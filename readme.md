# ZenZero - 中国象棋与五子棋 AI (ZenZero - Chinese Chess & Gomoku AI)

这是一个基于 AlphaZero 思想实现的通用棋类 AI 框架，目前已支持中国象棋（Chinese Chess）和五子棋（Gomoku）。项目采用了高度优化的架构，旨在充分利用多核 CPU 和 GPU 资源，实现高性能的自我对弈训练和对战。

## 🚀 核心特性 (Features)

* **高性能自我对弈:** 采用移除了 GIL 的 `free-threading` Python (`nogil`)，实现了真正并行的自我对弈数据生成。
* **动态批处理推理:** 独立的推理服务器，采用动态批处理（Dynamic Batching）技术，最大化 GPU 利用率。
* **Cython 核心加速:** 走法生成和胜负判断等核心逻辑使用 Cython 进行了 C 级别的优化。
* **AI 竞技场 (Arena):** 自动化的 AI 竞技场，通过对战更新 ELO 等级分，科学评估模型棋力。
* **前后端分离架构:**
    * **训练端:** 采用高效的本地 Unix Socket 进行低延迟通信。
    * **对战端:** 支持 AI 算力与 UI 界面分布式部署（例如：高性能服务器负责 MCTS 搜索，本地轻便设备仅负责图形交互）。
* **图形化对战界面 (GUI):** 基于 Pygame 实现，支持 Human vs. AI、AI vs. AI 等多种模式。
* **模块化设计:** 环境、MCTS、网络、播放器、UI 模块清晰分离，易于扩展。

## 📥 预训练权重 (Pre-trained Models)

本项目不直接在仓库中包含大体积模型文件。为了直接开始对战，请按以下步骤配置：

1.  前往 [Releases 页面](https://github.com/yadiel-tien/AlphaGomoku/releases) 下载对应游戏的 `checkpoint_xxx.pt` 文件。
2.  运行一次程序，系统将自动创建 `data/ChineseChess/` 和 `data/Gomoku/` 目录。
3.  将下载的 `.pt` 文件放入对应的游戏子目录下即可。
4.  **注意：** 系统会自动识别编号最大的 `checkpoint` 作为最强模型，您无需手动配置 `best_index.pkl`。

解压或放置后的目录结构示例：
```text
data/
├── ChineseChess/
│   └── checkpoint_498.pt
└── Gomoku/
    └── checkpoint_141.pt
```

## ⚙️ 项目结构 (Project Structure)

```
ZenZero/
├── core/               # 核心逻辑 (env, mcts, network, player, utils)
├── services/           # 后端服务 (inference, train)
├── apps/               # 应用层 (ui, assets)
├── scripts/            # 启动脚本
├── logs/               # 日志文件
└── data/               # 存放模型、棋谱等数据 (需手动创建)
```

## 🧩 扩展新游戏 (Extending to New Games)

ZenZero 旨在作为一个通用的 AlphaZero 框架。除了内置的象棋和五子棋，您可以轻松添加其他棋类游戏：

### 1. 实现环境 (Environment)
在 `core/env/` 目录下创建新的环境类，继承自 `BaseEnv`。您需要实现以下核心接口：
- `get_valid_actions()`: 获取当前合法走法。
- `step(action)`: 执行动作并返回新状态。
- `is_terminated()`: 判断比赛是否结束。
- `convert_to_network_input()`: 将棋盘状态转换为神经网络所需的 Tensor。

### 2. 配置参数 (Configuration)
在 `core/utils/config.py` 中为新游戏添加配置项（包括 `screen_size`, `grid_size`, `n_res_blocks` 等），并修改全局 `game_name`：
```python
CONFIG = {
    'game_name': 'YourNewGame',  # 在这里切换当前要训练/运行的游戏
    'YourNewGame': {
        # 具体的网络结构和游戏参数
    }
}
```

### 3. 实现 UI (可选)
在 `apps/ui/` 下参考 `chess.py` 或 `gomoku.py` 实现图形化界面。

## 🛠️ 安装与设定 (Setup & Installation)

本项目采用**算力分离**架构，建议准备两个独立的 Python 环境：

### 1. 创建标准环境 (用于训练、推理、UI)
用于运行 PyTorch 训练服务器、推理 Hub 以及图形界面。建议使用 Python 3.10+。
```bash
python3 -m venv standard_venv
source standard_venv/bin/activate
pip install torch numpy requests pygame flask
```

### 2. 创建 nogil 环境 (仅用于高性能自我对弈)
用于运行大规模并行的 MCTS 搜索。推荐使用 [uv](https://github.com/astral-sh/uv) 安装 free-threading 版本。
```bash
# 安装 3.14t 并创建环境
uv python install 3.14t
uv venv --python 3.14t nogil_venv
source nogil_venv/bin/activate
# 仅需安装基础库 (selfplay 通过 socket 与标准环境的服务器通信)
uv pip install numpy requests
```

### 3. 编译 Cython 模块
必须使用 `nogil` 环境编译以确保线程安全：
```bash
source nogil_venv/bin/activate
cd core/env
python setup.py build_ext --inplace
cd ../..
```

## 🎮 使用方法 (Usage)

### 1. 训练新模型
1. **指定游戏**: 在 `core/utils/config.py` 中修改 `game_name`。
2. **启动训练**:
    * **终端 1 (标准环境)**: `python -m scripts.train_server` (负责权重更新与推理)
    * **终端 2 (nogil 环境)**: `python -X gil=0 -m scripts.selfplay` (负责并行产生数据)

### 2. 图形化对战 (推荐)
1. **终端 1 (标准环境)**: `python -m scripts.play_server` (对战 API 服务，已集成推理调度功能)
2. **终端 2 (标准环境)**: `python -m scripts.ui_play` (GUI 界面)


### 3. AI 竞技场
```bash
source nogil_venv/bin/activate
python -m scripts.arena
```

## 📝 配置 (Configuration)
文件路径、模型超参数等均可在 `core/utils/config.py` 中修改。

## 📜 授权 (License)
MIT License.
