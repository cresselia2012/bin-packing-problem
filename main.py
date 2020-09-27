"""
1. ビンパッキング問題を生成
2. ビンパッキング問題をpulp(CBC)で求解
3. 結果を積み上げ棒グラフで可視化
"""

# ビンパッキング問題生成に必要
import random

# ビンパッキング問題求解に必要
import pulp

# 結果の可視化に必要
import pandas as pd
import matplotlib.pyplot as plt


def solve_bin_packing_problem(bin_size, item_size, threads):
    """ビンパッキング問題を解く"""

    num_items = len(item_size)

    # 問題をセット
    problem = pulp.LpProblem("BBP", pulp.LpMinimize)

    # 変数をセット
    var_x = [
        [pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(num_items)]
        for i in range(num_items)
    ]

    var_y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(num_items)]

    # 目的関数をセット
    problem += pulp.lpSum(y_i for y_i in var_y)

    # 制約条件をセット
    for x_i, y_i in zip(var_x, var_y):
        problem += pulp.lpDot(item_size, x_i) - bin_size * y_i <= 0

    for j in range(num_items):
        problem += pulp.lpSum(var_x[i][j] for i in range(num_items)) == 1

    # CBCを呼び出して解く
    result = problem.solve(pulp.PULP_CBC_CMD(msg=0, threads=threads))

    return result, problem, var_x, var_y

def output_graph(item_size, bins):
    """結果を積み上げ棒グラフで可視化する"""

    maxsize = max([len(bins_k) for bins_k in bins])
    dataset = pd.DataFrame(
        [
            [item_size[bins_k[i]] if i < len(bins_k) else 0 for bins_k in bins]
            for i in range(maxsize)
        ],
        columns=[f"{i}" for i in range(len(bins))],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(dataset)):
        ax.bar(dataset.columns, dataset.iloc[i], bottom=dataset.iloc[:i].sum())
    plt.show()
    #fig.savefig("result.png")

def main():
    """main method"""

    # ビンパッキング問題を生成する
    ## ビンの大きさ
    bin_size = 10

    ## アイテムの数
    num_items = 100

    ## 各アイテムの大きさを 1〜ビンの大きさの半分でランダムに生成
    item_size = [random.randint(1, bin_size / 2) for i in range(num_items)]

    # ビンパッキング問題を解く
    result, problem, var_x, var_y = solve_bin_packing_problem(bin_size, item_size, 8)

    # 結果を出力する
    if result == 1:
        optimal_value = pulp.value(problem.objective)
        print(f"optimal value: {optimal_value}")

        bins = []
        for i, (x_i, y_i) in enumerate(zip(var_x, var_y)):
            if pulp.value(y_i) == 1:
                bins.append([i for i, x_ij in enumerate(x_i) if pulp.value(x_ij) == 1])
                print(f"{i}: {bins[-1]}")

        # 結果を積み上げ棒グラフで可視化する
        output_graph(item_size, bins)
    else:
        print("PULP_CBC could not solve BBP")


if __name__ == "__main__":
    main()
