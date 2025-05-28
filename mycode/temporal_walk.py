import numpy as np
import globals

class Temporal_Walk(object):
    def __init__(self, learn_data, inv_relation_id, transition_distr):
        """
        Initialize temporal random walk object.

        Parameters:
            learn_data (np.ndarray): data on which the rules should be learned
            inv_relation_id (dict): mapping of relation to inverse relation
            transition_distr (str): transition distribution
                                    "unif" - uniform distribution
                                    "exp"  - exponential distribution

        Returns:
            None
        """

        self.learn_data = learn_data
        self.inv_relation_id = inv_relation_id
        self.transition_distr = transition_distr
        self.neighbors = store_neighbors(learn_data)
        self.edges = store_edges(learn_data)

    def sample_start_edge(self, rel_idx):
        """
        Define start edge distribution.

        Parameters:
            rel_idx (int): relation index

        Returns:
            start_edge (np.ndarray): start edge
        """

        rel_edges = self.edges[rel_idx]
        start_edge = rel_edges[np.random.choice(len(rel_edges))]

        return start_edge

    def sample_next_edge(self, filtered_edges, cur_ts):
        """
        Define next edge distribution.

        Parameters:
            filtered_edges (np.ndarray): filtered (according to time) edges
            cur_ts (int): current timestamp

        Returns:
            next_edge (np.ndarray): next edge
        """

        if self.transition_distr == "unif":
            next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
        elif self.transition_distr == "exp":
            if len(filtered_edges) > 1:
                rels = set(filtered_edges[:, 1])
                max_cur_rel = []
                max_num = 0
                for rel in rels:
                    num = len(filtered_edges[filtered_edges[:, 1] == rel])
                    if num > max_num:
                        max_num = num
                        max_cur_rel = [rel]
                    elif num == max_num:
                        max_cur_rel.append(rel)
                cur_rel = max_cur_rel[np.random.choice(len(max_cur_rel))]
                can_edges = filtered_edges[filtered_edges[:, 1] == cur_rel]
                next_edge = can_edges[np.random.choice(len(can_edges))]
            else:
                next_edge = filtered_edges[0]

            # tss = filtered_edges[:, 3]
            # prob = np.exp(tss - cur_ts)
            # try:
            #     prob = prob / np.sum(prob)
            #     next_edge = filtered_edges[
            #         np.random.choice(range(len(filtered_edges)), p=prob)
            #     ]
            # except ValueError:  # All timestamps are far away
            #     next_edge = filtered_edges[np.random.choice(len(filtered_edges))]

        return next_edge

    def transition_step(self, cur_node, cur_ts, prev_edge, start_node, ts, step, L, type):
        """
        Sample a neighboring edge given the current node and timestamp.
        In the second step (step == 1), the next timestamp should be smaller than the current timestamp.
        In the other steps, the next timestamp should be smaller than or equal to the current timestamp.
        In the last step (step == L-1), the edge should connect to the source of the walk (cyclic walk).
        It is not allowed to go back using the inverse edge.

        Parameters:
            cur_node (int): current node
            cur_ts (int): current timestamp
            prev_edge (np.ndarray): previous edge
            start_node (int): start node
            step (int): number of current step
            L (int): length of random walk

        Returns:
            next_edge (np.ndarray): next edge
        """
        next_edges = self.neighbors[cur_node]

        if step == 1:  # The next timestamp should be smaller than the current timestamp
            filtered_edges = next_edges[next_edges[:, 3] < cur_ts + globals.delta]
        else:  # The next timestamp should be smaller than or equal to the current timestamp
            if type == 'Head':
                filtered_edges = next_edges[(next_edges[:, 3] >= cur_ts - globals.delta) * (next_edges[:, 3] < ts + globals.delta)]
                inv_edge = prev_edge
            else:
                filtered_edges = next_edges[next_edges[:, 3] <= cur_ts + globals.delta]
                # Delete inverse edge
                inv_edge = [
                    cur_node,
                    self.inv_relation_id[prev_edge[1]],
                    prev_edge[0],
                    cur_ts,
                ]
            row_idx = np.where(np.all(filtered_edges == inv_edge, axis=1))
            filtered_edges = np.delete(filtered_edges, row_idx, axis=0)

        if (type == 'Head' and step == 1) or (type == 'Tail' and step == 1) or (type == 'Middle' and step == 2):  # Find an edge that connects to the source of the walk
            filtered_edges = filtered_edges[filtered_edges[:, 2] == start_node]

        if len(filtered_edges):
            next_edge = self.sample_next_edge(filtered_edges, cur_ts)
        else:
            next_edge = []

        return next_edge

    def sample_walk(self, L, rel_idx, type):
        """
        Try to sample a cyclic temporal random walk of length L (for a rule of length L-1).

        Parameters:
            L (int): length of random walk
            rel_idx (int): relation index

        Returns:
            walk_successful (bool): if a cyclic temporal random walk has been successfully sampled
            walk (dict): information about the walk (entities, relations, timestamps)
        """

        walk_successful = True
        walk = dict()
        prev_edge = self.sample_start_edge(rel_idx)
        start_node = prev_edge[0]
        cur_node = prev_edge[2]
        ts = prev_edge[3]
        cur_ts = prev_edge[3]
        walk["entities"] = [start_node, cur_node]
        walk["relations"] = [prev_edge[1]]
        walk["timestamps"] = [ts]

        for step in range(1, L):
            next_edge = self.transition_step(
                cur_node, cur_ts, prev_edge, start_node, ts, step, L, type
            )
            if len(next_edge):
                if type == 'Head':
                    if step != 1:
                        inv_edge = [
                            next_edge[2],
                            np.int64(self.inv_relation_id[next_edge[1]]),
                            next_edge[0],
                            next_edge[3],
                        ]
                        next_edge = inv_edge
                    cur_node = next_edge[0]
                else:
                    if type == 'Middle':
                        if step == 1:
                            inv_edge = [
                                next_edge[2],
                                np.int64(self.inv_relation_id[next_edge[1]]),
                                next_edge[0],
                                next_edge[3],
                            ]
                            next_edge = inv_edge
                    cur_node = next_edge[2]
                cur_ts = next_edge[3]
                if type =='Head':
                    walk["relations"].insert(1,next_edge[1])
                    if step == 1:
                        walk["entities"].insert(2, next_edge[2])
                    walk["entities"].insert(2, next_edge[0])
                    walk["timestamps"].insert(1, cur_ts)
                else:
                    walk["relations"].append(next_edge[1])
                    if type == 'Middle' and step == 1:
                        walk["entities"].append(next_edge[0])
                    walk["entities"].append(cur_node)
                    walk["timestamps"].append(cur_ts)
                prev_edge = next_edge
            else:  # No valid neighbors (due to temporal or cyclic constraints)
                walk_successful = False
                break

        return walk_successful, walk


def store_neighbors(quads):
    """
    Store all neighbors (outgoing edges) for each node.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        neighbors (dict): neighbors for each node
    """

    neighbors = dict()
    nodes = list(set(quads[:, 0]))
    for node in nodes:
        neighbors[node] = quads[quads[:, 0] == node]

    return neighbors


def store_edges(quads):
    """
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]

    return edges
