
import pandas as pd

a = {1:2,3:4}

def func(dict):
    dict[1] = 1000

print(a)
func(a)
print(a)


df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],

                    'value': [1, 2, 3, 5]})

df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo'],

                    'value': [5, 6, 7, 8]})


df3 = pd.DataFrame()
print(pd.concat([df1,df3]).reset_index(drop=True))