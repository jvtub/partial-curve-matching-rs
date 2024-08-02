from partial_curve_matching import Vector, partial_curve

ps = [Vector(0,0), Vector(1,1), Vector(2,2)]
qs = [Vector(-1,-1), Vector(3,3)]

assert partial_curve(ps, qs, 0.1) != None # ps is a subcurve of qs
assert partial_curve(qs, ps, 0.1) == None # qs is not a subcurve of ps

qs2= [Vector(1,-1), Vector(5,3)] # Translated to the right.
assert partial_curve(ps, qs2, 0.1) == None

result = partial_curve(ps, qs, 0.1)
if result != None:
    # Subcurve result consists of point index + offset along qs.
    (t1, t2) = result
    print(t1, t2) # qs consists of two points (with index 0 and 1), so t1 and t2 will be in range [0, 1].

