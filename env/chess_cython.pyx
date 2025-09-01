# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from libc.stdint cimport int32_t
cimport numpy as cnp

ctypedef cnp.int32_t INT32_t

# ----------------------
# 棋子合法移动函数
# ----------------------


cdef int add_rook_dest(INT32_t * board, int r, int c, INT32_t *result) nogil:
    # 预分配足够大的numpy数组（假设最多20个移动）
    cdef int count = 0

    cdef int dr, dc, nr, nc, target, i
    cdef int directions[4][2]
    directions[0][0] = -1
    directions[0][1] = 0
    directions[1][0] = 1
    directions[1][1] = 0
    directions[2][0] = 0
    directions[2][1] = -1
    directions[3][0] = 0
    directions[3][1] = 1

    cdef bint own_side = board[r * 9 + c] < 7

    for i in range(4):
        dr = directions[i][0]
        dc = directions[i][1]
        nr = r
        nc = c
        while True:
            nr += dr
            nc += dc
            if nr < 0 or nr >= 10 or nc < 0 or nc >= 9:
                break
            target = board[nr * 9 + nc]
            if target == -1 or (target < 7) != own_side:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = nr
                result[count * 4 + 3] = nc
                count += 1
            if target != -1:
                break

    # 返回实际有效的结果部分
    return count

cdef int add_horse_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """马的合法移动"""
    cdef int count = 0

    cdef int nr, nc, br, bc, target, i
    cdef int horse_moves[8][4]
    horse_moves[0][0] = -2
    horse_moves[0][1] = -1
    horse_moves[0][2] = -1
    horse_moves[0][3] = 0
    horse_moves[1][0] = -2
    horse_moves[1][1] = 1
    horse_moves[1][2] = -1
    horse_moves[1][3] = 0
    horse_moves[2][0] = -1
    horse_moves[2][1] = -2
    horse_moves[2][2] = 0
    horse_moves[2][3] = -1
    horse_moves[3][0] = -1
    horse_moves[3][1] = 2
    horse_moves[3][2] = 0
    horse_moves[3][3] = 1
    horse_moves[4][0] = 1
    horse_moves[4][1] = -2
    horse_moves[4][2] = 0
    horse_moves[4][3] = -1
    horse_moves[5][0] = 1
    horse_moves[5][1] = 2
    horse_moves[5][2] = 0
    horse_moves[5][3] = 1
    horse_moves[6][0] = 2
    horse_moves[6][1] = -1
    horse_moves[6][2] = 1
    horse_moves[6][3] = 0
    horse_moves[7][0] = 2
    horse_moves[7][1] = 1
    horse_moves[7][2] = 1
    horse_moves[7][3] = 0

    cdef bint own_side = board[r * 9 + c] < 7

    for i in range(8):
        nr = r + horse_moves[i][0]
        nc = c + horse_moves[i][1]
        br = r + horse_moves[i][2]
        bc = c + horse_moves[i][3]

        if nr < 0 or nr >= 10 or nc < 0 or nc >= 9:
            continue
        if board[br * 9 + bc] != -1:  #蹩马腿
            continue
        target = board[nr * 9 + nc]
        if target == -1 or (target < 7) != own_side:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = nr
            result[count * 4 + 3] = nc
            count += 1

    return count

cdef int add_bishop_dest(INT32_t * board, int r, int c, INT32_t *result) nogil:
    """相的合法移动"""
    cdef int count = 0

    cdef int moves[32 * 4]
    moves[0 * 4 + 0] = 0
    moves[0 * 4 + 1] = 2
    moves[0 * 4 + 2] = 2
    moves[0 * 4 + 3] = 0
    moves[1 * 4 + 0] = 0
    moves[1 * 4 + 1] = 2
    moves[1 * 4 + 2] = 2
    moves[1 * 4 + 3] = 4
    moves[2 * 4 + 0] = 0
    moves[2 * 4 + 1] = 6
    moves[2 * 4 + 2] = 2
    moves[2 * 4 + 3] = 4
    moves[3 * 4 + 0] = 0
    moves[3 * 4 + 1] = 6
    moves[3 * 4 + 2] = 2
    moves[3 * 4 + 3] = 8
    moves[4 * 4 + 0] = 2
    moves[4 * 4 + 1] = 0
    moves[4 * 4 + 2] = 0
    moves[4 * 4 + 3] = 2
    moves[5 * 4 + 0] = 2
    moves[5 * 4 + 1] = 0
    moves[5 * 4 + 2] = 4
    moves[5 * 4 + 3] = 2
    moves[6 * 4 + 0] = 2
    moves[6 * 4 + 1] = 4
    moves[6 * 4 + 2] = 0
    moves[6 * 4 + 3] = 2
    moves[7 * 4 + 0] = 2
    moves[7 * 4 + 1] = 4
    moves[7 * 4 + 2] = 4
    moves[7 * 4 + 3] = 2
    moves[8 * 4 + 0] = 2
    moves[8 * 4 + 1] = 4
    moves[8 * 4 + 2] = 0
    moves[8 * 4 + 3] = 6
    moves[9 * 4 + 0] = 2
    moves[9 * 4 + 1] = 4
    moves[9 * 4 + 2] = 4
    moves[9 * 4 + 3] = 6
    moves[10 * 4 + 0] = 2
    moves[10 * 4 + 1] = 8
    moves[10 * 4 + 2] = 0
    moves[10 * 4 + 3] = 6
    moves[11 * 4 + 0] = 2
    moves[11 * 4 + 1] = 8
    moves[11 * 4 + 2] = 4
    moves[11 * 4 + 3] = 6
    moves[12 * 4 + 0] = 4
    moves[12 * 4 + 1] = 2
    moves[12 * 4 + 2] = 2
    moves[12 * 4 + 3] = 0
    moves[13 * 4 + 0] = 4
    moves[13 * 4 + 1] = 2
    moves[13 * 4 + 2] = 2
    moves[13 * 4 + 3] = 4
    moves[14 * 4 + 0] = 4
    moves[14 * 4 + 1] = 6
    moves[14 * 4 + 2] = 2
    moves[14 * 4 + 3] = 4
    moves[15 * 4 + 0] = 4
    moves[15 * 4 + 1] = 6
    moves[15 * 4 + 2] = 2
    moves[15 * 4 + 3] = 8
    moves[16 * 4 + 0] = 9
    moves[16 * 4 + 1] = 2
    moves[16 * 4 + 2] = 7
    moves[16 * 4 + 3] = 0
    moves[17 * 4 + 0] = 9
    moves[17 * 4 + 1] = 2
    moves[17 * 4 + 2] = 7
    moves[17 * 4 + 3] = 4
    moves[18 * 4 + 0] = 9
    moves[18 * 4 + 1] = 6
    moves[18 * 4 + 2] = 7
    moves[18 * 4 + 3] = 4
    moves[19 * 4 + 0] = 9
    moves[19 * 4 + 1] = 6
    moves[19 * 4 + 2] = 7
    moves[19 * 4 + 3] = 8
    moves[20 * 4 + 0] = 7
    moves[20 * 4 + 1] = 0
    moves[20 * 4 + 2] = 9
    moves[20 * 4 + 3] = 2
    moves[21 * 4 + 0] = 7
    moves[21 * 4 + 1] = 0
    moves[21 * 4 + 2] = 5
    moves[21 * 4 + 3] = 2
    moves[22 * 4 + 0] = 7
    moves[22 * 4 + 1] = 4
    moves[22 * 4 + 2] = 9
    moves[22 * 4 + 3] = 2
    moves[23 * 4 + 0] = 7
    moves[23 * 4 + 1] = 4
    moves[23 * 4 + 2] = 5
    moves[23 * 4 + 3] = 2
    moves[24 * 4 + 0] = 7
    moves[24 * 4 + 1] = 4
    moves[24 * 4 + 2] = 9
    moves[24 * 4 + 3] = 6
    moves[25 * 4 + 0] = 7
    moves[25 * 4 + 1] = 4
    moves[25 * 4 + 2] = 5
    moves[25 * 4 + 3] = 6
    moves[26 * 4 + 0] = 7
    moves[26 * 4 + 1] = 8
    moves[26 * 4 + 2] = 9
    moves[26 * 4 + 3] = 6
    moves[27 * 4 + 0] = 7
    moves[27 * 4 + 1] = 8
    moves[27 * 4 + 2] = 5
    moves[27 * 4 + 3] = 6
    moves[28 * 4 + 0] = 5
    moves[28 * 4 + 1] = 2
    moves[28 * 4 + 2] = 7
    moves[28 * 4 + 3] = 0
    moves[29 * 4 + 0] = 5
    moves[29 * 4 + 1] = 2
    moves[29 * 4 + 2] = 7
    moves[29 * 4 + 3] = 4
    moves[30 * 4 + 0] = 5
    moves[30 * 4 + 1] = 6
    moves[30 * 4 + 2] = 7
    moves[30 * 4 + 3] = 4
    moves[31 * 4 + 0] = 5
    moves[31 * 4 + 1] = 6
    moves[31 * 4 + 2] = 7
    moves[31 * 4 + 3] = 8

    cdef int r2, c2, tr, tc, target, i
    cdef bint own_side = board[r * 9 + c] < 7

    for i in range(32):
        r2 = moves[i * 4 + 0]
        c2 = moves[i * 4 + 1]
        tr = moves[i * 4 + 2]
        tc = moves[i * 4 + 3]
        if r2 != r or c2 != c:
            continue
        if board[(r + tr) // 2 * 9 + (c + tc) // 2] != -1:  #蹩象眼
            continue
        target = board[tr * 9 + tc]
        if target == -1 or (target < 7) != own_side:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = tr
            result[count * 4 + 3] = tc
            count += 1

    return count

cdef int add_advisor_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """士的合法移动"""
    cdef int count = 0

    cdef int moves[16][4]
    moves[0][0] = 0
    moves[0][1] = 3
    moves[0][2] = 1
    moves[0][3] = 4
    moves[1][0] = 0
    moves[1][1] = 5
    moves[1][2] = 1
    moves[1][3] = 4
    moves[2][0] = 1
    moves[2][1] = 4
    moves[2][2] = 0
    moves[2][3] = 3
    moves[3][0] = 1
    moves[3][1] = 4
    moves[3][2] = 0
    moves[3][3] = 5
    moves[4][0] = 1
    moves[4][1] = 4
    moves[4][2] = 2
    moves[4][3] = 3
    moves[5][0] = 1
    moves[5][1] = 4
    moves[5][2] = 2
    moves[5][3] = 5
    moves[6][0] = 2
    moves[6][1] = 3
    moves[6][2] = 1
    moves[6][3] = 4
    moves[7][0] = 2
    moves[7][1] = 5
    moves[7][2] = 1
    moves[7][3] = 4
    moves[8][0] = 9
    moves[8][1] = 3
    moves[8][2] = 8
    moves[8][3] = 4
    moves[9][0] = 9
    moves[9][1] = 5
    moves[9][2] = 8
    moves[9][3] = 4
    moves[10][0] = 8
    moves[10][1] = 4
    moves[10][2] = 9
    moves[10][3] = 3
    moves[11][0] = 8
    moves[11][1] = 4
    moves[11][2] = 9
    moves[11][3] = 5
    moves[12][0] = 8
    moves[12][1] = 4
    moves[12][2] = 7
    moves[12][3] = 3
    moves[13][0] = 8
    moves[13][1] = 4
    moves[13][2] = 7
    moves[13][3] = 5
    moves[14][0] = 7
    moves[14][1] = 3
    moves[14][2] = 8
    moves[14][3] = 4
    moves[15][0] = 7
    moves[15][1] = 5
    moves[15][2] = 8
    moves[15][3] = 4

    cdef int i, target
    cdef bint own_side = board[r * 9 + c] < 7

    for i in range(16):
        if moves[i][0] != r or moves[i][1] != c:
            continue
        target = board[moves[i][2] * 9 + moves[i][3]]
        if target == -1 or (target < 7) != own_side:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = moves[i][2]
            result[count * 4 + 3] = moves[i][3]
            count += 1

    return count

cdef int add_king_dest(INT32_t * board, int r, int c, INT32_t *result) nogil:
    """将/帅的合法移动"""
    cdef int count = 0

    cdef int moves[4][2]
    moves[0][0] = 1
    moves[0][1] = 0
    moves[1][0] = -1
    moves[1][1] = 0
    moves[2][0] = 0
    moves[2][1] = 1
    moves[3][0] = 0
    moves[3][1] = -1

    cdef int i, to_r, to_c, target
    cdef bint is_red = board[r * 9 + c] == 4
    cdef int dr, nr, rival_king

    for i in range(4):
        to_r = r + moves[i][0]
        to_c = c + moves[i][1]
        if to_r < 0 or to_r >= 10 or to_c < 0 or to_c >= 9:
            continue
        if is_red and to_r < 7:
            continue
        if not is_red and to_r > 2:
            continue
        if to_c < 3 or to_c > 5:
            continue
        target = board[to_r * 9 + to_c]
        if target == -1 or (target < 7) != is_red:
            result[count * 4 + 0] = r
            result[count * 4 + 1] = c
            result[count * 4 + 2] = to_r
            result[count * 4 + 3] = to_c
            count += 1

    # 两帅相对

    if 3 <= c <= 5:
        dr = -1 if is_red else 1
        nr = r + dr
        rival_king = 11 if is_red else 4

        while 0 <= nr < 10:
            target = board[nr * 9 + c]
            if target == -1:
                nr += dr
            elif target == rival_king:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = nr
                result[count * 4 + 3] = c
                count += 1
                break
            else:
                break

    return count

cdef int add_cannon_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """炮的合法移动"""
    cdef int count = 0

    cdef int directions[4][2]
    directions[0][0] = -1
    directions[0][1] = 0
    directions[1][0] = 1
    directions[1][1] = 0
    directions[2][0] = 0
    directions[2][1] = -1
    directions[3][0] = 0
    directions[3][1] = 1

    cdef int dr, dc, nr, nc, target, i
    cdef bint own_side = board[r * 9 + c] < 7
    cdef bint found_screen

    for i in range(4):
        dr = directions[i][0]
        dc = directions[i][1]
        nr = r + dr
        nc = c + dc
        found_screen = False

        while 0 <= nr < 10 and 0 <= nc < 9:
            target = board[nr * 9 + nc]
            if not found_screen:
                if target == -1:
                    result[count * 4 + 0] = r
                    result[count * 4 + 1] = c
                    result[count * 4 + 2] = nr
                    result[count * 4 + 3] = nc
                    count += 1
                else:
                    found_screen = True
            else:
                if target != -1:
                    if (target < 7) != own_side:
                        result[count * 4 + 0] = r
                        result[count * 4 + 1] = c
                        result[count * 4 + 2] = nr
                        result[count * 4 + 3] = nc
                        count += 1
                    break
            nr += dr
            nc += dc

    return count

cdef int add_pawn_dest(INT32_t * board, int r, int c, INT32_t * result) nogil:
    """卒/兵合法移动"""
    cdef int count = 0
    cdef int to_r, to_c, target, dc
    cdef bint is_red = board[r * 9 + c] < 7  # 假设红方棋子编号 < 7

    # 1. 前进一步
    if is_red:
        to_r = r - 1
        to_c = c
        if to_r >= 0:
            target = board[to_r * 9 + to_c]
            if target == -1 or (target < 7) != is_red:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = to_r
                result[count * 4 + 3] = to_c
                count += 1
    else:
        to_r = r + 1
        to_c = c
        if to_r < 10:
            target = board[to_r * 9 + to_c]
            if target == -1 or (target < 7) != is_red:
                result[count * 4 + 0] = r
                result[count * 4 + 1] = c
                result[count * 4 + 2] = to_r
                result[count * 4 + 3] = to_c
                count += 1

    # 2. 过河后可以左右走
    if (is_red and r <= 4) or (not is_red and r >= 5):

        for dc in range(2):  # -1,1
            to_r = r
            to_c = c + <int>((-1) ** dc)
            if 0 <= to_c < 9:
                target = board[to_r * 9 + to_c]
                if target == -1 or (target < 7) != is_red:
                    result[count * 4 + 0] = r
                    result[count * 4 + 1] = c
                    result[count * 4 + 2] = to_r
                    result[count * 4 + 3] = to_c
                    count += 1

    return count

def get_valid_actions(cnp.ndarray[INT32_t, ndim=3, mode='c'] state,
                      int player_to_move,
                      cnp.ndarray[INT32_t, ndim=4, mode='c'] move2action):
    """Cython版本的合法动作获取"""
    cdef int count = 0

    cdef cnp.ndarray[INT32_t, ndim=2, mode='c'] board_2d = np.ascontiguousarray(state[:, :, 0])
    cdef int32_t * move2action_ptr = <int32_t *> move2action.data
    cdef int32_t * board_ptr = <int32_t *> board_2d.data
    cdef int r, c, piece, action_id, dest_num, to_r, to_c, i
    cdef int32_t total_buffer[200], result_buffer[20 * 4]

    with nogil:
        for r in range(10):
            for c in range(9):
                piece = board_ptr[r * 9 + c]
                if piece == -1:
                    continue

                # 检查棋子属于当前玩家
                if (player_to_move == 0 and 0 <= piece <= 6) or (player_to_move == 1 and piece >= 7):
                    # 获取该棋子的所有目标位置
                    dest_num = get_destinations(piece, board_ptr, r, c, &result_buffer[0])

                    # 将每个移动转换为动作ID
                    for i in range(dest_num):
                        to_r = result_buffer[i * 4 + 2]
                        to_c = result_buffer[i * 4 + 3]

                        action_id = move2action_ptr[r * 810 + c * 90 + to_r * 9 + to_c]

                        total_buffer[count] = action_id
                        count += 1

    cdef cnp.ndarray[INT32_t, ndim=1] result = np.empty(count, dtype=np.int32)
    for i in range(count):
        result[i] = total_buffer[i]

    return result

cdef int get_destinations(int piece, INT32_t * board, int r, int c, INT32_t *result) nogil:
    if piece == 0 or piece == 7:
        return add_rook_dest(board, r, c, result)
    elif piece == 1 or piece == 8:
        return add_horse_dest(board, r, c, result)
    elif piece == 2 or piece == 9:
        return add_bishop_dest(board, r, c, result)
    elif piece == 3 or piece == 10:
        return add_advisor_dest(board, r, c, result)
    elif piece == 4 or piece == 11:
        return add_king_dest(board, r, c, result)
    elif piece == 5 or piece == 12:
        return add_cannon_dest(board, r, c, result)
    else:
        return add_pawn_dest(board, r, c, result)
