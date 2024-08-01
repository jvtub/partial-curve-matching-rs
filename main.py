from partial_curve_matching import Vector, partial_curve

ps = [Vector(0,0), Vector(1,1), Vector(2,2)]
qs = [Vector(-1,-1), Vector(3,3)]
result = partial_curve(ps, qs, 0.1)
