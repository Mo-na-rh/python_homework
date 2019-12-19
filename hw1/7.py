#even indexes
data = list(input().split())

n = len(data)

for i in range(n):
    if i%2==0:
        print(data[i])

