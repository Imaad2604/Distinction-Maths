A_dok = {}

A_dok[(0, 0)] = 10
A_dok[(0, 3)] = 2
A_dok[(1, 0)] = 3
A_dok[(1, 1)] = 9
A_dok[(2, 1)] = 7
A_dok[(2, 2)] = 8
A_dok[(2, 3)] = 7
A_dok[(3, 3)] = 5


print(A_dok.get((1, 1), 0)) 
print(A_dok.get((3, 0), 0)) 