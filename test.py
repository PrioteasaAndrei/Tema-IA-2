
a = {1:2,3:4}

def func(dict):
    dict[1] = 1000

print(a)
func(a)
print(a)