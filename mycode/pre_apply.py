import json
import time
import argparse
import itertools
import mpmath
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12


def sort_rules_dict(rules_dict):
    for rel in rules_dict:
        rules_dict[rel] = sorted(
            rules_dict[rel], key=lambda x: (x["conf"], x["rule_body_supp"]), reverse=True
        )
    return rules_dict

def save_rules(rules_dict):
    rules_dict = {int(k): v for k, v in rules_dict.items()}
    filename = "pre_rules.json"
    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(rules_dict, fout)

def update_rules_dict(rules_dict, rule):
    try:
        rules_dict[rule["head_rel"]].append(rule)
    except KeyError:
        rules_dict[rule["head_rel"]] = [rule]
    return rules_dict

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="icews14", type=str)
parser.add_argument("--test_data", default="valid", type=str)
parser.add_argument("--rules", "-r", default="090425051520_r[1,2,3]_n200_exp_s20_rules.json", type=str)
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

def apply_rules(i, num_queries):
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
    all_rules = []

    num_rest_queries = len(test_data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(test_data))

    cur_ts = test_data[test_queries_idx[0]][3]
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    it_start = time.time()
    for j in test_queries_idx:
        test_query = test_data[j]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                cands_dict = [dict() for _ in range(len(args))]
                walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            if test_query[2] in dict(itertools.islice(cands_dict[s].items(), 10)):
                                all_rules.append(rule)
        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()
    return all_rules


start = time.time()
num_queries = len(test_data) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)
end = time.time()

final_all_rules = dict()
for i in range(num_processes):
    for rule in output[i]:
        update_rules_dict(final_all_rules, rule)
    output[i].clear()

total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))

rules_dict = sort_rules_dict(final_all_rules)

rules_statistics(final_all_rules)

save_rules(final_all_rules)
