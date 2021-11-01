import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_physical_network(net, labels=True):
    edge_density = {}
    node_density = {
        l.name : len(l.coder) for l in net.layers.values()
    }

    for l in net.layers.values():
        for conn in l.connections.values():
            k = (conn.to_layer.name, conn.from_layer.name)
            edge_density[k] = edge_density.get(k, 0) + len(conn.learned)

    for g in net.gates:
        if g[1] == "context":
            print(g)
            edge_density[g[0],g[2]] = 1

    edges = [(fl,tl) for (tl,fl) in edge_density.keys()]
    print("\nNodes:")
    for n in node_density:
        print(n)
    print("\nEdges:")
    #for e in sorted(edges, key= lambda x:(x[1],x[0])):
    for e in sorted(edges, key= lambda e:edge_density.get(e,0)):
        print("%5d: %10s -> %-10s" % ((edge_density.get(e,0),) + e))
    print()

    g = nx.DiGraph()
    g.add_edges_from(edges)
    ##pos=nx.spring_layout(g,weight=1,iterations=100)
    #pos=nx.kamada_kawai_layout(g)
    #pos=nx.fruchterman_reingold_layout(g, pos=pos)
    ##pos=nx.planar_layout(g)
    ##pos=nx.circular_layout(g)

    pos=nx.fruchterman_reingold_layout(g)

    pos=nx.kamada_kawai_layout(g, pos=pos)

    node_colors = [
        'red' if "_ctx" in l else 'green'
        for l in g.nodes()]

    for (fl,tl) in edges:
        if "_ctx" in fl:
            edge_density[(tl,fl)] = node_density[fl]

    edge_widths = [edge_density.get((tl,fl), 0) for (fl,tl) in g.edges()]
    edge_widths = [5 * w / max(edge_widths) for w in edge_widths]
    edge_widths = [w+0.5 for w in edge_widths]

    node_sizes = [node_density.get(tl, 0) for tl in g.nodes()]
    node_sizes = [300 * w / max(node_sizes) for w in node_sizes]

    if labels:
        nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(g, pos, alpha=1.0, width=edge_widths)

    plt.show()

def draw_virtual_network(net, layer_grid=None, mix_layers=False,
        include_flashed=True, labels=True):
    mappings = {
        conn.name : [(v.tuple()[1],k[1])
                for k,v in conn.mappings.items()
                    if include_flashed or (k not in conn.flashed_mappings)]
            for l in net.layers.values()
                for conn in l.connections.values()
#                    if conn.to_layer.name != "gh" or conn.from_layer.name in ("op", "gh")
    }

    if layer_grid is None:
        layer_grid = [["mem", "lex", "op", "gh"]]
    only_layers = [l for row in layer_grid for l in row if l is not None]

    edges = []
    for (tl,fl,name),m in mappings.items():
        if only_layers is not None and any(
            l not in only_layers for l in (tl,fl)): continue

        for syms in m:
            tsym, fsym = syms[:2]
            fnode = ("%s_%s" % (fl,fsym))
            tnode = ("%s_%s" % (tl,tsym))
            if fsym != tsym or tl != fl:
                edges.append((fl,tl,fnode,tnode))

    layers = only_layers if only_layers is not None else set(
        a for a,b,c,d in edges).union(set(b for a,b,c,d in edges))
    colors = ["red", "blue", "green", "orange",
              "purple", "brown", "black", "yellow"]
    color_map = {}

    poss = {}
    gs = {}
    for j,ls in enumerate(reversed(layer_grid)):
        for i,to_layer in enumerate(ls):
            g = nx.DiGraph()
            nodes = set(c for a,b,c,d in edges if a == to_layer).union(
                set(d for a,b,c,d in edges if b == to_layer))
            g.add_nodes_from(nodes)
            g.add_edges_from([(c,d)
                for a,b,c,d in edges
                    if a == to_layer and b == to_layer])

            color_map[to_layer] = colors[len(color_map)%len(colors)]

            print("%10s %5d %s" % (to_layer, len(nodes), color_map[to_layer]))
            #for node in sorted(nodes):
            #    print(node)

            if len(nodes) > 0:
                d = tuple(d for n,d in g.degree())
                std_d = np.std(d)
                if std_d != 0.:
                    it = int(20. * (np.mean(d) / np.std(d)))
                else:
                    it = 20

                try:
                    pos=nx.planar_layout(g)
                    pos=nx.fruchterman_reingold_layout(g, pos=pos)
                except:
                    #pos=nx.kamada_kawai_layout(g,weight=0.5)
                    try:
                        pos=nx.fruchterman_reingold_layout(g)
                        pos=nx.spring_layout(g,weight=1, pos=poss)
                    except:
                        pos=nx.kamada_kawai_layout(g,weight=0.5)
                #pos=nx.spring_layout(g,weight=0.5,iterations=it,pos=pos)
                #pos=nx.circular_layout(g)
                gs[to_layer] = g

                if len(nodes) == 1:
                    for k,v in pos.items():
                        poss[k] = v + np.array([
                            coeff_x*i, coeff_y*j])
                elif len(nodes) > 1:
                    xs = [v[0] for v in pos.values()]
                    ys = [v[1] for v in pos.values()]
                    x_min,y_min = (min(xs),min(ys))
                    x_range, y_range = max(xs)-x_min, max(ys)-y_min

                    def renorm(v):
                        return np.array((
                            (v[0] - x_min) / x_range,
                            (v[1] - y_min) / y_range))

                    coeff_x=1.5
                    coeff_y=1.5

                    for k,v in pos.items():
                        poss[k] = renorm(v) + np.array([
                            coeff_x*i, coeff_y*j])

    g = nx.DiGraph()

    # Update node positions based on inter-layer edges
    if mix_layers:
        g.add_edges_from(((c,d) for a,b,c,d in edges))
        gd = {n:d for n,d in g.degree()}

        for h in gs.values():
            poss.update(nx.kamada_kawai_layout(g,pos=poss))
            #poss.update(nx.spring_layout(g,weight=1, pos=poss))
            #poss.update(nx.fruchterman_reingold_layout(g, pos=poss))

    # Draw layer graphs
    for to_layer,h in gs.items():
        node_colors = color_map[to_layer]

        degrees = [len(tuple(1 for a,b,c,d in edges if n in (c,d)))
                for n in h.nodes()]

        node_size = [300 * (d/max(degrees)) for d in degrees]
        if labels:
            nx.draw_networkx_labels(h, poss, font_size=8)
        nx.draw_networkx_nodes(h, poss, node_color=node_colors, node_size=node_size)
        nx.draw_networkx_edges(h, poss, alpha=1.0, width=0.1)

    # Draw inter-layer connections
    for to_layer in layers:
        e = [(c,d) for a,b,c,d in edges if a != b and b == to_layer]
        if len(e) > 0:
            g.add_edges_from(e)
    for to_layer in layers:
        e = [(c,d) for a,b,c,d in edges if a != b and b == to_layer]
        if len(e) > 0:
            nx.draw_networkx_edges(g, poss,
                edge_color=color_map[to_layer], alpha=0.25, edgelist=e)

    plt.show()

