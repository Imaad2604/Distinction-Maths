A = [
    [10, 0, 0, 2],
    [3, 9, 0, 0],
    [0, 7, 8, 7],
    [0, 0, 0, 5]
]


row = []
col = []
data = []

for i in range(len(A)):          
    for j in range(len(A[0])):   
        if A[i][j] != 0:         
            row.append(i)
            col.append(j)
            data.append(A[i][j])

print("row  =", row)
print("col  =", col)
print("data =", data)