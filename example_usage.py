from partial_curve_matching import Vector, partial_curve, make_graph, partial_curve_graph
from random import random
from math import sqrt

# Partial curve matching against a curve.
print("Partial curve matching against a curve.")
ps = [Vector(0,0), Vector(1,1), Vector(2,2)]
qs = [Vector(-1,-1), Vector(3,3), Vector(5, 5)]

assert partial_curve(ps, qs, 0.1) != None # ps is a subcurve of qs
assert partial_curve(qs, ps, 0.1) == None # qs is not a subcurve of ps

qs2 = [Vector(1,-1), Vector(5,3), Vector(7, 5)] # Translate qs to the right.
assert partial_curve(ps, qs2, 0.1) == None 
assert partial_curve(ps, qs2, 2) != None  # Sufficiently high threshold results in a match.

# When partial matching, first check a result is found, 
result = partial_curve(ps, qs, 0.1) 
if result != None: # Check a valid subcurve is found.
    # If a subcurve exists, print the subcurve (interval 0 to 1) of qs.
    print(result)


# Partial curve matching against a graph.

# Constructing an example graph (basically a grid with high connectivity).
print("Partial curve matching against a graph.")
size = 100
vertices = [((size*y + x), Vector(x, y)) for x in range(size) for y in range(size)]
edges = [] 
for y in range(size):
    for x in range(size):
        if x < size - 1: # Link to right.
            edges.append((size*y+x, size*y+(x+1)))
        if y < size - 1: # Link to above.
            edges.append((size*y+x, size*(y+1)+x))
        if x < size - 1 and y < size - 1: # Link to diagonal right.
            edges.append((size*y+x, size*(y+1)+(x+1)))
        if x > 0 and y < size - 1: # Link to diagonal left.
            edges.append((size*y+x, size*(y+1)+(x-1)))
G = make_graph(vertices, edges)

# Generate some curve to seek partial match in.
# ps = [Vector(0, 0), Vector(size - 1, size - 1)] # Success
# ps = [Vector(0, 0), Vector(1, 1)] # Success
# ps = [Vector(0, 0), Vector(10, 10)] # Success

# Move diagonally right, move down, move diagonally left.
eps = 0.2 
deviation = sqrt(0.5 * eps * eps) * 0.99
noise = lambda: 2 * (random() - 0.5) * deviation
ps = [(0, 0), (size - 1, size - 1), (size - 1, 0), (0, size - 1)] 
ps = list(map(lambda v: Vector(v[0] + noise(), v[1] + noise()), ps))

result = partial_curve_graph(G, ps, 0.2)
if result != None:
    print(result)