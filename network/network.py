import os
from typing import Self

import torch
from torch import Tensor, nn

from inference.functions import get_model_path, get_checkpoint_path
from utils.config import CONFIG
from utils.types import EnvName


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int) -> None:
        """特征提取，输入[B,C,H,W],输出[B,n_filters,H,W]"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # 如果输入是[C,H,W],则补上B=1，->[1,C,H,W]
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block,自调节通道权重，可用于增强网络性能。"""
    """暂时还没用"""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, n_filters: int, with_se: bool = False) -> None:
        """残差块处理特征，输入输出结构一样[B,n_filters,H,W]"""
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        # 可选SEBlock
        self.with_se = with_se
        if with_se:
            self.se = SEBlock(n_filters)

        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += x  # 残差连接
        if self.with_se:
            y = self.se(y)
        return self.relu2(y)


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, n_cells: int, n_actions: int) -> None:
        """
        策略头，输出动作策略概率分布。[B,in_channels,H,W]->[B,H*W]/[H*W]
        :param in_channels: 输入通道数
        :param n_filters: 中间特征通道数量
        :param n_actions: 动作空间大小
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(n_cells * n_filters, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        # x:[B,in_channels,H,W]
        x = self.relu(self.bn(self.conv(x)))
        # x:[B,n_filters,H,W]
        x = x.reshape(x.shape[0], -1)  # 展平
        # x:[B,n_filters*H*W]
        x = self.fc(x)
        # x:[B,n_actions]
        # 直接返回原始 Logits
        return x


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, n_cells: int) -> None:
        """价值头,输出当前盘面价值。【B,in_channels,H,W]->[B]"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.relu1 = nn.LeakyReLU()
        self.fc1 = nn.Linear(n_cells, hidden_channels)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # x:[B,in_channels,H,W]
        x = self.conv(x)
        x = self.relu1(x)
        # x:[B,1,H,W]
        x = x.reshape(x.shape[0], -1)
        # x:[B,H*W]
        x = self.fc1(x)
        x = self.fc2(self.relu2(x))
        # x:[B,1]
        value = self.tanh(x).reshape(-1)
        # x:[B]
        return value


class Net(nn.Module):
    def __init__(self, n_filters: int, n_cells=15 * 15, n_res_blocks=7, n_channels=2, n_actions=15 * 15,
                 with_se: bool = False) -> None:
        """
        类alpha zero结构，双头输出policy和value
        :param n_filters: 卷积层通道数
        :param n_cells: 动作空间大小，应等于H*W
        :param n_res_blocks: 残差块个数
        """
        super().__init__()
        self.conv_block = ConvBlock(n_channels, n_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_filters, with_se) for _ in range(n_res_blocks)])
        self.policy = PolicyHead(n_filters, 32, n_cells, n_actions)
        self.value = ValueHead(n_filters, 256, n_cells)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.policy(x), self.value(x)

    @classmethod
    def make_raw_model(cls, env_name: EnvName, eval_model: bool, with_se: bool = True) -> Self:
        """工厂方法创建初始模型"""
        settings = CONFIG[env_name]
        model = cls(settings['n_filter'], settings['n_cells'], settings['n_res_blocks'],
                    settings['n_channels'], settings['n_actions'], with_se).to(CONFIG['device'])
        if eval_model:
            model.eval()
            print('Raw model created in eval mode.')
        else:
            print('Raw model created in training mode.')
        return model

    def load_from_index(self, model_idx: int, env_name: EnvName) -> bool:
        """尝试通过id加载模型，返回加载结果"""
        # 先尝试从模型加载
        model_path = get_model_path(env_name, model_idx)
        if os.path.exists(model_path):
            self.load_state_dict(
                torch.load(model_path, map_location=CONFIG['device']))
            print(f'Model {model_idx} loaded successfully from {model_path}.')
            return True
        # 尝试从存档加载
        checkpoint_path = get_checkpoint_path(env_name, model_idx)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
            self.load_state_dict(checkpoint['model'])
            print(f'Model {model_idx} loaded successfully from {checkpoint_path}.')
            return True
        return False
