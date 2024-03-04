a = [1,2,3]
def extend(a):
    a.extend([1,2,3])
    return True
if extend(a):
    print(111)
print(a)