# AlphaZero 在中国象棋与五子棋的实现 (AlphaZero for Chinese Chess and Gomoku)

这是一个基于AlphaZero思想实现的通用棋类AI框架，目前已支持中国象棋（Chinese Chess）和五子棋（Gomoku）。项目采用了高度优化的架构，旨在充分利用多核CPU和GPU资源，实现高性能的自我对弈训练和对战。

## 🚀 核心特性 (Features)

* **高性能自我对弈:** 采用移除了GIL的`free-threading` Python (`nogil`)，结合`ThreadPoolExecutor`，实现了真正并行的自我对弈数据生成。
* **动态批处理推理:** 独立的推理服务器，采用动态批处理（Dynamic Batching）技术，能根据请求压力自动调整批次大小，最大化GPU利用率。
* **Cython核心加速:** 游戏的核心逻辑，如走法生成和胜负判断，已使用Cython进行了C级别的优化，极大地提升了计算速度。
* **AI竞技场 (Arena):** 内建一个自动化的AI竞技场，可以让不同版本的模型进行对战，并根据结果更新其ELO等级分，用于科学地评估模型棋力的进展。
* **前后端分离架构:**
    * **训练端:** 采用高效的本地Unix Socket进行低延迟通信。
    * **对战端:** 提供了基于Flask的Web API服务器，可以轻松地与各种客户端（如网页、桌面应用）进行对战。
* **图形化对战界面 (GUI):** 基于Pygame实现了功能完善的图形化界面，支持人类 vs. AI、AI vs. AI多种对战模式。
* **模块化设计:** 项目的各个部分（环境、MCTS、网络、播放器、UI）都进行了清晰的模块化分离，易于维护和扩展。

## ⚙️ 项目结构 (Project Structure)

```
five_in_a_row/
├── env/                # 游戏环境定义 (chess.py, gomoku.py, chess_cython.pyx)
├── inference/          # 推理服务器相关 (client.py, engine.py, hub.py, server.py)
├── mcts/               # MCTS 算法实现 (deepMcts.py)
├── network/            # 神经网络模型定义 (network.py)
├── player/             # 玩家抽象 (human.py, ai_client.py, ai_server.py)
├── scripts/            # 可执行的主脚本
├── train/              # 训练与自我对弈管理器 (selfplay.py)
├── ui/                 # Pygame UI 实现
├── utils/              # 通用工具 (config.py, replay.py, elo.py)
└── data/               # 存放模型、棋谱等数据 (需手动创建)
```

## 🛠️ 安装与设定 (Setup & Installation)

本项目需要两个独立的Python环境：一个用于高性能计算的`nogil`环境，一个用于UI的标准Python环境。

1.  **克隆项目:**
    ```bash
    git clone git@github.com:yadiel-tien/AlphaGomoku.git
    cd five_in_a_row
    ```

2.  **创建`nogil`计算环境 (用于训练和推理):**
    * 请确保您已经从源代码编译好了`free-threading` (nogil) 版本的Python（例如 Python 3.14）。
    * 创建虚拟环境：
        ```bash
        /path/to/your/nogil_python -m venv nogil_venv
        source nogil_venv/bin/activate
        ```
    * 安装依赖：
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt  # 建议您创建一个requirements.txt文件
        ```

3.  **创建标准UI环境 (用于图形界面):**
    * 使用您系统的标准Python创建虚拟环境：
        ```bash
        python3 -m venv standard_venv
        source standard_venv/bin/activate
        ```
    * 安装依赖 (主要是 `pygame` 和 `requests`):
        ```bash
        pip install pygame requests
        ```

4.  **编译Cython模块:**
    * 已含编译后so文件，可直接使用
    * **必须**使用`nogil`环境来编译，以确保线程安全。
    * 首先，切换虚拟环境：
        ```bash
        source nogil_venv/bin/activate
        ```
    * 然后执行编译：
        ```bash
        cd env
        python setup.py build_ext --inplace
        cd ..
        ```

## 🎮 使用方法 (Usage)

所有指令都在项目根目录下执行。

### 1. 训练新模型 (自我对弈 + 训练)

这需要同时开启两个终端窗口，并且都**启用`nogil_venv`环境**。

* **在【终端1】，启动训练与推理服务器:**
    ```bash
    source nogil_venv/bin/activate
    python scripts/train_server.py
    ```
    服务器会开始运行，等待来自自我对弈客户端的连接和训练指令。

* **在【终端2】，启动自我对弈客户端:**
    ```bash
    source nogil_venv/bin/activate
    # 使用 -X gil=0 标志来彻底禁用GIL
    python -X gil=0 scripts/selfplay.py
    ```
    自我对弈程序会开始高速生成棋局数据，并在每轮结束后通知服务器进行训练。
### 2. 命令行对战 (Human vs. AI)

这需要两个终端，分别在**不同**的Python环境中运行。

* **在【终端1 - 管理中心】，启动Hub服务器:**
    * Hub负责动态创建和管理推理服务。
    * 在**标准Python环境**环境中运行。
    ```bash
    conda active fiveInARow
    python scripts/infer_hub.py # 假设这是您的Hub启动脚本
  ```
* **在【终端2 - 命令行对战界面】，启动AI对战API服务器:**
    * 这个服务器负责接收UI的请求并进行MCTS计算。
    * 在**标准Python环境**环境中运行或 **`nogil_venv`环境**都可以。
    ```bash
    conda active fiveInARow
    python scripts/play_server.py # 这是您的Flask服务器
    ```


### 3. 图形化对战 (Human vs. AI)

这需要两个终端，分别在**不同**的Python环境中运行。

* **在【终端1 - 管理中心】，启动Hub服务器:**
    * Hub负责动态创建和管理推理服务。
    * 在**标准Python环境**环境中运行。
    ```bash
    conda active fiveInARow
    python scripts/infer_hub.py # 假设这是您的Hub启动脚本
  ```
* **在【终端2 - 远程后端服务】，启动AI对战API服务器:**
    * 这个服务器负责接收UI的请求并进行MCTS计算。通过http服务传输，支持局域网不同主机通信。
    * 在**标准Python环境**环境中运行。。
    ```bash
    conda active fiveInARow
    python scripts/play_server.py # 这是您的Flask服务器
    ```

* **在【终端3 - 游戏界面】，启动Pygame UI客户端:**
    * 这个程序只负责图形界面，不进行复杂计算。
    * **必须**在能稳定运行Pygame的**标准Python环境**中运行。
    ```bash
    conda active fiveInARow
    python scripts/ui_play.py # 这是您的Pygame主程序
    ```
    Pygame窗口将会启动，您可以与通过API服务器运行的AI进行对战。

### 4. AI竞技场 (评估模型棋力)

* **准备工作:** 在`data/rates/ChineseChess/` (或Gomoku) 目录下，创建一个 `candidates.txt` 文件，里面用逗号分隔写上您想评估的模型ID，例如：`450,460,470,480`。

* **启动竞技场:**
    * 竞技场会并行地进行多场AI对战，也需要在`nogil_venv`环境中运行。
    ```bash
    source nogil_venv/bin/activate
    python scripts/run_arena.py # 假设您有一个启动Arena的脚本
    ```

## 📝 设置 (Configuration)

项目的主要参数，如文件路径、模型超参数等，都可以在 `utils/config.py` 中进行修改。

## 📜 授权 (License)

MIT License。

---