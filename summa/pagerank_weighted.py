from scipy.sparse import csr_matrix
from scipy.linalg import eig
from numpy import empty as empty_matrix
import igraph as ig

try:
    from numpy import VisibleDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

CONVERGENCE_THRESHOLD = 0.0001


def pagerank_weighted(graph, initial_value=None, damping=0.85):
    """Calculates PageRank for an undirected graph"""
    if initial_value == None: initial_value = 1.0 / len(graph.nodes())
    scores = dict.fromkeys(graph.nodes(), initial_value)

    iteration_quantity = 0
    for iteration_number in range(100):
        iteration_quantity += 1
        convergence_achieved = 0
        for i in graph.nodes():
            rank = 1 - damping
            for j in graph.neighbors(i):
                neighbors_sum = sum(graph.edge_weight((j, k)) for k in graph.neighbors(j))
                rank += damping * scores[j] * graph.edge_weight((j, i)) / neighbors_sum

            if abs(scores[i] - rank) <= CONVERGENCE_THRESHOLD:
                convergence_achieved += 1

            scores[i] = rank

        if convergence_achieved == len(graph.nodes()):
            break

    return scores


def pagerank_weighted_scipy(graph, damping=0.85):
    g = ig.Graph()
    print(graph.nodes())
    node_to_int_mapping = dict([(n, i) for (i, n) in enumerate(graph.nodes())])
    edges_int = [(node_to_int_mapping[n1], node_to_int_mapping[n2]) for (n1, n2) in graph.edge_properties]
    edges_weitghts = [graph.edge_weight(edge) for edge in graph.edge_properties]

    g.add_vertices(node_to_int_mapping.values())
    g.add_edges(edges_int)

    ranking = g.pagerank(weights=edges_weitghts, damping=damping, niter=100)
    return dict([(n, ranking[node_to_int_mapping[n]]) for n in graph.nodes()])


def build_adjacency_matrix(graph):
    row = []
    col = []
    data = []
    nodes = graph.nodes()
    length = len(nodes)

    for i in range(length):
        current_node = nodes[i]
        neighbors_sum = sum(graph.edge_weight((current_node, neighbor)) for neighbor in graph.neighbors(current_node))
        for j in range(length):
            edge_weight = float(graph.edge_weight((current_node, nodes[j])))
            if i != j and edge_weight != 0:
                row.append(i)
                col.append(j)
                data.append(edge_weight / neighbors_sum)

    return csr_matrix((data,(row,col)), shape=(length,length))


def build_probability_matrix(graph):
    dimension = len(graph.nodes())
    matrix = empty_matrix((dimension,dimension))

    probability = 1 / float(dimension)
    matrix.fill(probability)

    return matrix


def process_results(graph, vecs):
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vecs[i][0])

    return scores
