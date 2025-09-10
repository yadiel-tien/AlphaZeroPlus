import numpy as np
from numpy.typing import NDArray
import random


def random_mirror_state_ip(state: NDArray, env_name: str) -> tuple[NDArray, int]:
    """对 state 进行随机对称变换（旋转或翻转），会原地更改state

    参数：
        state: 棋盘状态张量，board_shape 为 (H, W, C)
        idx: 对称操作索引（0~7）：
       - 0: 原样
       - 1~3: 顺时针旋转 90°/180°/270°
       - 4: 左右镜像
       - 5: 上下镜像
       - 6: 主对角线翻转
       - 7: 副对角线翻转
        env_name: 游戏名称，控制支持的变换范围

    返回：
        变换后的 ndarray"""
    if env_name == 'ChineseChess':
        # 象棋只支持上下左右翻转
        if random.random() < 0.5 :
            return apply_symmetry(state, 4), 4

        return state, 0

    if env_name == 'Gomoku':
        shape = state.shape
        if shape[0] != shape[1]:
            indices = (0, 2, 4, 5)
        else:
            indices = range(8)
        idx = random.choice(indices)
        return apply_symmetry(state, idx), idx

    raise ValueError('Unknown game environment')


def apply_symmetry(array: NDArray, idx: int) -> NDArray:
    """ 都是返回副本
     :param array: 二维ndarray
        :param idx: 对称操作索引（0~7）：
       - 0: 原样
       - 1~3: 顺时针旋转 90°/180°/270°
       - 4: 左右镜像
       - 5: 上下镜像
       - 6: 主对角线翻转
       - 7: 副对角线翻转"""
    if array.ndim == 3:
        axes = (1, 0, 2)
    elif array.ndim == 2:
        axes = (1, 0)
    else:
        raise ValueError('array must be 2D or 3D')

    if idx == 0:
        return array.copy()
    elif idx < 4:  # 旋转 0°, 90°, 180°, 270°
        return np.rot90(array, k=idx)
    elif idx == 4:  # 水平翻转
        return np.fliplr(array)
    elif idx == 5:  # 垂直翻转
        return np.flipud(array)
    elif idx == 6:  # 主对角线翻转，转置
        return np.transpose(array, axes=axes)
    elif idx == 7:  # 副对角线翻转,先旋转180度，再转置
        rotated = np.rot90(array, k=2)
        return np.transpose(rotated, axes=axes)
    else:
        raise ValueError('Invalid index')


def mirror_board_policy(policy: NDArray, idx: int, board_shape: tuple[int, ...]) -> NDArray:
    """适用于动作空间跟二维棋盘一一对应的情况"""
    if board_shape[0] != board_shape[1] and idx in (1, 3, 6, 7):
        raise ValueError("Symmetry functions require a square board (rows == cols).")

    reshaped = policy.reshape(board_shape[:2])
    return apply_symmetry(reshaped, idx).flatten()


def mirror_action_policy(policy: NDArray, idx: int, lr_map: NDArray, ud_map: NDArray) -> NDArray:
    """动作空间跟棋盘平面无法映射，直接提供动作镜像map，根据map翻转。reverse仍可使用本方法"""
    if idx == 0:
        return policy
    elif idx == 4:
        # lr_map[i] = 镜像后 action 的索引
        # 反映为：镜像后的位置 i ← 原位置 lr_map[i]
        symmetry_policy = np.zeros_like(policy)
        symmetry_policy[lr_map] = policy
        return symmetry_policy
    elif idx == 5:
        symmetry_policy = np.zeros_like(policy)
        symmetry_policy[ud_map] = policy
        return symmetry_policy
    else:
        raise ValueError('Invalid index')


def reverse_board_policy(policy: NDArray, idx: int, board_shape: tuple[int, ...]) -> NDArray:
    """反转概率分布,概率可以跟棋盘平面一一对应"""
    if board_shape[0] != board_shape[1] and idx in (1, 3, 6, 7):
        raise ValueError("Symmetry functions require a square board (rows == cols).")

    reshaped = policy.reshape(board_shape[:2])
    if idx < 4:  # 逆向旋转
        return np.rot90(reshaped, k=-idx).flatten()
    elif idx == 4:  # 水平翻转
        return np.fliplr(reshaped).flatten()
    elif idx == 5:  # 垂直翻转
        return np.flipud(reshaped).flatten()
    elif idx == 6:  # 主对角线翻转
        return reshaped.T.flatten()
    elif idx == 7:  # 副对角线翻转 先转置再旋转
        return np.rot90(reshaped.T, k=2).flatten()
    else:
        raise ValueError(f'Invalid symmetry index: {idx}')


def test_apply_symmetry():
    # 构造简单 2D 棋盘（Gomoku policy 示例）
    board_2d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # 构造简单 3D 棋盘（state 示例，最后一维为通道）
    board_3d = np.stack([board_2d, board_2d + 10], axis=-1)  # board_shape = (3, 3, 2)
    transformed_list = []
    print("=== Testing 2D transformations ===")
    for idx in range(8):
        try:
            ud_transformed = apply_symmetry(board_2d, idx)
            print(f"idx {idx}:\n{ud_transformed}\n")
            transformed_list.append(ud_transformed)
        except Exception as e:
            print(f"2D failed at idx={idx}: {e}")

    print("=== Testing 3D transformations ===")
    for idx in range(8):
        try:
            ud_transformed = apply_symmetry(board_3d, idx)
            print(f"idx {idx}:\n{ud_transformed[:, :, 0]}\n")  # 打印第一个通道
            print(f"{ud_transformed[:, :, 1]}\n")  # 打印第二个通道
        except Exception as e:
            print(f"3D failed at idx={idx}: {e}")

    print("=== Testing 2D reverse transformations ===")
    for idx in range(8):
        try:
            ud_transformed = transformed_list[idx].flatten()
            reversed_arr = reverse_board_policy(ud_transformed, idx, (3, 3))
            print(f"idx {idx}:\n{reversed_arr.reshape((3, 3))}\n")
        except Exception as e:
            print(f"3D failed at idx={idx}: {e}")

    print("=== Testing action policy transformations ===")
    from env.chess import ChineseChess
    ChineseChess.init_class_dicts()
    policy = np.array(list(range(2086)))
    ud_transformed = mirror_action_policy(policy, 5, ChineseChess.mirror_lr_actions, ChineseChess.mirror_ud_actions)
    ud_reversed_policy = mirror_action_policy(ud_transformed, 5, ChineseChess.mirror_lr_actions,
                                              ChineseChess.mirror_ud_actions)
    lr_transformed = mirror_action_policy(policy, 4, ChineseChess.mirror_lr_actions, ChineseChess.mirror_ud_actions)
    lr_reversed_policy = mirror_action_policy(ud_transformed, 4, ChineseChess.mirror_lr_actions,
                                              ChineseChess.mirror_ud_actions)
    for action in range(50, 60):
        r, c, to_r, to_c = ChineseChess.action2move(action)
        print(f'action {action},move:({r},{c},{to_r},{to_c})')
        mirror_lr_action = ChineseChess.mirror_lr_actions[action]
        mirror_lr_move = ChineseChess.action2move(int(mirror_lr_action))
        mirror_ud_action = ChineseChess.mirror_ud_actions[action]
        mirror_ud_move = ChineseChess.action2move(int(mirror_ud_action))
        print(f'lr_action={mirror_lr_action},lr_move={mirror_lr_move},value={lr_transformed[mirror_lr_action]}')
        print(f'ud_action={mirror_ud_action},lr_move={mirror_ud_move},value={ud_transformed[mirror_ud_action]}')
        print(f'lr_reversed_policy_value={lr_reversed_policy[action]}')
        print(f'ud_reversed_policy_value={ud_reversed_policy[action]}')


if __name__ == "__main__":
    test_apply_symmetry()
