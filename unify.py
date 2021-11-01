from networkx import bfs_tree
from networkx.generators.trees import random_tree
from random import sample, choice, randint
from copy import deepcopy

def lispify_tree(tree_map, labels, curr_node):
    if len(tree_map[curr_node]) > 0:
        return "( %s )" % " ".join(
            lispify_tree(tree_map, labels, child)
            for child in tree_map[curr_node])
    else:
        return labels[curr_node]

def gen_tree(tree_size=15, syms="ABCDEFGHIJ"):
    graph = random_tree(tree_size)
    root = max(range(len(graph.nodes)), key = lambda i : len(graph.edges(i)))

    tree = bfs_tree(graph, root)
    labels = { i : choice(syms) for i in range(tree_size) }

    tree_map = { i : [] for i in range(tree_size) }

    for edge in tree.edges:
        parent, child = edge
        tree_map[parent].append(child)

    return tree_map, labels, root

def gen_trees(tree_size=15, syms="abcdefghij", var_names="VWXYZ",
        num_subs=5, max_var_size=3, mismatch=False):
    tree_map1, labels1, root = gen_tree(tree_size, syms)
    tree_map2, labels2 = deepcopy(tree_map1), deepcopy(labels1)

    leaves = [i for i,children in tree_map1.items() if len(children) == 0]
    num_subs = min(num_subs, len(leaves))
    subs = { }
    var_leaves = []

    for leaf in sample(leaves, num_subs):
        var_name = choice(var_names)
        var_leaves.append(leaf)

        if var_name not in subs:
            var_size = randint(1, max_var_size)
            subs[var_name] = lispify_tree(*gen_tree(var_size, syms))

        if choice((True,False)):
            labels1[leaf] = "( var %s )" % var_name
            labels2[leaf] = subs[var_name]
        else:
            labels2[leaf] = "( var %s )" % var_name
            labels1[leaf] = subs[var_name]

    if mismatch:
        t,l = choice([(tree_map1, labels1), (tree_map2, labels2)])
        mangle_node = choice(list(set(range(len(labels1))) - set(var_leaves)))
        l[mangle_node] = choice(
            list(set(syms) - set(l[mangle_node])))
        t[mangle_node] = []


    # TODO:
    # add parameter for whether trees should match
    # implement several possible mutations
    #   - insert/remove/modify leaf/tree
    return (lispify_tree(tree_map1, labels1, root),
        lispify_tree(tree_map2, labels2, root),
        subs)
