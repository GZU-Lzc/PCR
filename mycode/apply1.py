import json
import time
import argparse
import itertools
import numpy as np

import mpmath
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="icews14", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="280525020939_r[1,2,3]_n175_exp_s12_rules.json", type=str)
parser.add_argument("--rule_lengths", "-l", default=[1,2,3], type=int, nargs="+")
parser.add_argument("--window", "-w", default=0, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=128, type=int)
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths

dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
data = Grapher(dataset_dir)
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
rules_statistics(rules_dict)
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_rule_body_supp=2, rule_lengths=rule_lengths
)
print("Rules statistics after pruning:")
rules_statistics(rules_dict)
learn_edges = store_edges(data.train_idx)

score_func = score_12
# It is possible to specify a list of list of arguments for tuning
args = [[0.1, 0.5]]
mpmath.mp.dps = 16

def apply_rules(i, sub, obj, num_queries):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    if i < num_processes - 1:
        test_queries_idx = range(i * num_queries + sub, (i + 1) * num_queries + sub)
    else:
        test_queries_idx = range(i * num_queries + sub, obj)
    cur_ts = test_data[test_queries_idx[0]][3]
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    it_start = time.time()
    for j in test_queries_idx:
        test_query = test_data[j]
        cands_dict1 = [dict() for _ in range(len(args))]
        cands_dict2 = [dict() for _ in range(len(args))]
        cands_dict3 = [dict() for _ in range(len(args))]
        flag1, flag2, flag3 = 0, 0, 0

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        if flag1 and flag2 and flag3:
                            break
                        if rule['type'] == 'Head' and not flag1:
                            cands_dict1 = ra.get_candidates(
                                rule,
                                rule_walks,
                                cur_ts,
                                cands_dict1,
                                score_func,
                                args,
                                dicts_idx,
                            )
                            for s in dicts_idx:
                                cands_dict1[s] = {
                                    x: sorted(cands_dict1[s][x], reverse=True)
                                    for x in cands_dict1[s].keys()
                                }
                                cands_dict1[s] = dict(
                                    sorted(
                                        cands_dict1[s].items(),
                                        key=lambda item: item[1],
                                        reverse=True,
                                    )
                                )
                                top_k_scores1 = [v for _, v in cands_dict1[s].items()][:top_k]
                                unique_scores1 = list(
                                    scores for scores, _ in itertools.groupby(top_k_scores1)
                                )
                                if len(unique_scores1) >= top_k and top_k != 0:
                                    flag1 = 1
                        elif rule['type'] == 'Middle' and not flag2:
                            cands_dict2 = ra.get_candidates(
                                rule,
                                rule_walks,
                                cur_ts,
                                cands_dict2,
                                score_func,
                                args,
                                dicts_idx,
                            )
                            for s in dicts_idx:
                                cands_dict2[s] = {
                                    x: sorted(cands_dict2[s][x], reverse=True)
                                    for x in cands_dict2[s].keys()
                                }
                                cands_dict2[s] = dict(
                                    sorted(
                                        cands_dict2[s].items(),
                                        key=lambda item: item[1],
                                        reverse=True,
                                    )
                                )
                                top_k_scores2 = [v for _, v in cands_dict2[s].items()][:top_k]
                                unique_scores2 = list(
                                    scores for scores, _ in itertools.groupby(top_k_scores2)
                                )
                                if len(unique_scores2) >= top_k and top_k != 0:
                                    flag2 = 1
                        elif rule['type'] == 'Tail' and not flag3:
                            cands_dict3 = ra.get_candidates(
                                rule,
                                rule_walks,
                                cur_ts,
                                cands_dict3,
                                score_func,
                                args,
                                dicts_idx,
                            )
                            for s in dicts_idx:
                                cands_dict3[s] = {
                                    x: sorted(cands_dict3[s][x], reverse=True)
                                    for x in cands_dict3[s].keys()
                                }
                                cands_dict3[s] = dict(
                                    sorted(
                                        cands_dict3[s].items(),
                                        key=lambda item: item[1],
                                        reverse=True,
                                    )
                                )
                                top_k_scores3 = [v for _, v in cands_dict3[s].items()][:top_k]
                                unique_scores3 = list(
                                    scores for scores, _ in itertools.groupby(top_k_scores3)
                                )
                                if len(unique_scores3) >= top_k and top_k != 0:
                                    flag3 = 1
                        else:
                            continue

            cands_dict = [dict() for _ in range(len(args))]
            all_keys = set()
            for d in [cands_dict1[0], cands_dict2[0], cands_dict3[0]]:
                all_keys.update(d.keys())

            for key in all_keys:
                combined_values = []
                for d in [cands_dict1[0], cands_dict2[0], cands_dict3[0]]:
                    if key in d:
                        combined_values.extend(d[key])
                if combined_values:
                    cands_dict[0][key] = 1 - np.prod(1 - np.array(combined_values))
            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    # scores = []
                    # for values in cands_dict[s].values():
                    #     # 将得分列表转换为 mpmath 的浮点数类型
                    #     values_mp = [mpmath.mpf(float(score)) for score in values]
                    #     # 计算 1 - score 的乘积
                    #     product = mpmath.fprod([mpmath.mpf(1) - score for score in values_mp])
                    #     # 计算 noisy - OR 结果
                    #     noisy_or_score = mpmath.mpf(1) - product
                    #     scores.append(noisy_or_score)

                    scores = list(
                        map(
                            lambda x: np.max(np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    del cands_dict
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        if not (j - test_queries_idx[0] + 1) % 10:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

    return all_candidates, no_cands_counter


result_total_time = 0
result_final_no_cands_counter = 0
for k in range(0, 8):
    print('第', k + 1, '/8轮开始\n')
    length = int(len(test_data) / 8)
    if k != 7:
        sub = k * length
        obj = (k + 1) * length
    else:
        sub = k * length
        obj = len(test_data)
    l = obj - sub
    num_queries = l // num_processes
    start = time.time()
    output = Parallel(n_jobs=num_processes)(
        delayed(apply_rules)(i, sub, obj, num_queries) for i in range(num_processes)
    )
    end = time.time()

    final_all_candidates = [dict() for _ in range(len(args))]
    for s in range(len(args)):
        for i in range(num_processes):
            final_all_candidates[s].update(output[i][0][s])
            output[i][0][s].clear()

    final_no_cands_counter = 0
    for i in range(num_processes):
        final_no_cands_counter += output[i][1]

    total_time = round(end - start, 6)
    result_total_time = result_total_time + total_time
    result_final_no_cands_counter += final_no_cands_counter
    print("Application finished in {} seconds.".format(total_time))
    print("No candidates: ", final_no_cands_counter, " queries")
    print("Final application finished in {} seconds.".format(result_total_time))
    print("Final no candidates: ", result_final_no_cands_counter, " queries")
    print("\n")

    for s in range(len(args)):
        score_func_str = score_func.__name__ + str(args[s])
        score_func_str = score_func_str.replace(" ", "")
        ra.save_candidates(
            rules_file,
            dir_path,
            final_all_candidates[s],
            rule_lengths,
            window,
            score_func_str,
        )
