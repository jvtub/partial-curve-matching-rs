from partial_curve_matching import Vector, partial_curve_graph_linear, make_linear_graph

p1 = Vector(0, 0)
p2 = Vector(0.5, 0.5)
p3 = Vector(1, 1)

curve = [p1, p3]

vertices = [(0, p1), (1, p2), (2, p3)]
edges = [(0, 1), (1, 2)]
graph = make_linear_graph(vertices, edges)

result = partial_curve_graph_linear(graph, curve, 0.01)
result = partial_curve_graph_linear(graph, [p3, p1], 0.01)
result = partial_curve_graph_linear(graph, [Vector(0.3, 0.3), Vector(0.6, 0.6), Vector(0.8, 0.8)], 0.01)
result = partial_curve_graph_linear(graph, [Vector(0.3, 0.3), Vector(0.4, 0.4), Vector(0.45, 0.45)], 0.01)
result = partial_curve_graph_linear(graph, [Vector(0.3, 0.3), Vector(0.2, 0.2)], 0.01)
