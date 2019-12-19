# dictionares
# votes in usa

n = int(input())
d = dict()
for i in range(n):
    fam , result = input().split()
    d[fam] = d.get(fam,0) + int(result)
for fam, result in sorted(d.items()):
    print(fam, result)

