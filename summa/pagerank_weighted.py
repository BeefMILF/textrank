import igraph as ig

try:
    from numpy import VisibleDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

CONVERGENCE_THRESHOLD = 0.0001


def pagerank_weighted(graph, damping=0.85):

    node_to_int_mapping = dict([(n, i) for (i, n) in enumerate(graph.nodes())])
    edges_weights = [graph.edge_weight(edge) for edge in graph.edge_properties]
    g = setup_igraph(graph, node_to_int_mapping)

    ranking = g.pagerank(weights=edges_weights, damping=damping, niter=100)
    return process_results(ranking, graph.nodes(), node_to_int_mapping)


def process_results(ranking, nodes, node_to_int_mapping):
    return dict([(n, ranking[node_to_int_mapping[n]]) for n in nodes])


def setup_igraph(graph, node_to_int_mapping):
    g = ig.Graph()
    edges_int = [(node_to_int_mapping[n1], node_to_int_mapping[n2]) for (n1, n2) in graph.edge_properties]

    g.add_vertices(node_to_int_mapping.values())
    g.add_edges(edges_int)
    return g