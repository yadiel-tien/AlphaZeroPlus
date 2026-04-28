def rating1v1(a, b, score_a=0.0):
    # 玩家 A 的预期得分
    e_a = 1 / (1 + 10 ** ((b.scores - a.scores) / 400))

    # 玩家 B 的实际得分
    score_b = 1 - score_a
    # 玩家 B 的预期得分
    e_b = 1 - e_a
    # 计算新评分
    a.scores += a.k * (score_a - e_a)
    b.scores += b.k * (score_b - e_b)


class Elo:
    def __init__(self, index, initial_score=1500.0):
        self.index = index
        self.scores = initial_score
        self.n_games_played = 0
        self.records = {}  # {rival_index:[win,lose,draw]}

    @property
    def k(self):
        """根据对弈局数动态获取 K 值"""
        if self.n_games_played < 50:  # 假设前50局是新手期
            return 32
        elif self.n_games_played < 200:  # 50-200局是中等活跃期
            return 24
        else:  # 200局以上是稳定期
            return 16

    def defeat(self, rival):
        self.n_games_played += 1
        rival.n_games_played += 1
        if rival.index not in self.records:
            self.records[rival.index] = [0] * 3
        self.records[rival.index][0] += 1
        if self.index not in rival.records:
            rival.records[self.index] = [0] * 3
        rival.records[self.index][1] += 1
        rating1v1(self, rival, 1)

    def draw(self, rival):
        self.n_games_played += 1
        rival.n_games_played += 1
        if rival.index not in self.records:
            self.records[rival.index] = [0] * 3
        self.records[rival.index][2] += 1
        if self.index not in rival.records:
            rival.records[self.index] = [0] * 3
        rival.records[self.index][2] += 1
        rating1v1(self, rival, 0.5)

    def __str__(self):
        return f'index:{self.index},score:{self.scores:.2f},games:{self.n_games_played}'

    def show_records(self):
        for k, v in self.records.items():
            print(f'VS {k},win:{v[0]},loss:{v[1]},draw:{v[2]},win_rate:{v[0] / sum(v):.2%}.')
