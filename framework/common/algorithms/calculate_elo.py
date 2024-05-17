#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import sys
import os


def expected(A, B):
    """
    Calculate expected score of A in a match against B
    :param A: Elo rating for player A
    :param B: Elo rating for player B
    """
    return 1 / (1 + 10 ** ((B - A) / 400))


def elo_update(exp, score, k=32):
    """
    Calculate the new Elo rating for a player
    :param old: The previous Elo rating
    :param exp: The expected score for this match
    :param score: The actual score for this match: win 1.0, draw 0.5, loss 0
    :param k: The k-factor for Elo (default: 32)
    """
    return k * (score - exp)

def cal_elo(winrate_file, fix_model_file = 'fixModel/fix_score_model_label.txt', iteration_num = 10000):
    res = []
    fix_model_dict = {}
    if os.path.exists(fix_model_file):
        with open(fix_model_file, 'r') as fr:
            for line in fr:
                arr = line.strip().split(':')
                fix_model_dict[arr[1].strip()] = float(arr[2].strip())
    with open(winrate_file, 'r') as fr:
        for line in fr:
            items = line.strip().split(',')
            camp1_version = items[0].strip()
            camp2_version = items[1].strip()
            camp1_win = int(items[3].strip())
            draw = int(items[4].strip())
            camp1_loss = int(items[5].strip())
            res.append([camp1_version, camp2_version, camp1_win, draw, camp1_loss])
    result_dict = {}
    init_score = 1000
    for line in res:
        player1 = line[0]
        player2 = line[1]
        if fix_model_dict.has_key(player1):
            result_dict[player1] = fix_model_dict[player1]
        else:
            result_dict[player1] = init_score
        if fix_model_dict.has_key(player2):
            result_dict[player2] = fix_model_dict[player2]
        else:
            result_dict[player2] = init_score
    for i in range(iteration_num):
        update_dict = {}
        for key in result_dict.keys():
            update_dict[key] = []
        for line in res:
            camp1_version, camp2_version, camp1_win, draw, camp1_loss = line
            camp1_score = (1.0 * camp1_win + 0.5 * draw) / (camp1_win + draw + camp1_loss)
            camp1_update = elo_update(expected(result_dict[camp1_version], result_dict[camp2_version]),
                                      camp1_score, k=32)
            update_dict[camp1_version].append(camp1_update)
            update_dict[camp2_version].append(-camp1_update)

        for key in result_dict.keys():
            if not fix_model_dict.has_key(key):
                result_dict[key] += np.mean(update_dict[key])
    return result_dict

if __name__ == '__main__':
    result = cal_elo(sys.argv[1])
    for item in result.items():
        print(f"{item[0]}:{item[1]}")