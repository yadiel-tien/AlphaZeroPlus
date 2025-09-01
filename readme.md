## This is an AlphaZero like Dual-Game AI Framework
```mermaid
graph LR
  subgraph SelfPlay 客户端
    SP[自我博弈循环]
    MCTS[MCTS + PUCT]
    InferClient[推理客户端 (socket)]
    Buffer[经验缓存 (state, π, z/q)]
    SP --> MCTS
    MCTS --> InferClient
    InferClient --> InferenceServer
    MCTS --> Buffer
  end

  subgraph 推理服务
    InferenceServer[推理服务]
    InferQueue[推理队列]
    NN[神经网络前向推理]
    InferenceServer --> InferQueue --> NN --> InferenceServer
  end

  subgraph 训练进程
    Dataset[训练数据池]
    Trainer[Trainer]
    Net[神经网络 (Conv+Res+Head)]
    EMA[EMA 平滑]
    Save[模型保存与评估]
    Dataset --> Trainer
    Trainer --> Net
    Trainer --> EMA
    Net --> Trainer
    Trainer --> Save
  end

  InferenceServer -. IPC通信 .-> InferenceServer
```