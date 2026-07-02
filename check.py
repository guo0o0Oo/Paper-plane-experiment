import pandas as pd
import re

df = pd.read_excel('result_final.xlsx')
def parse_id(exp_id):
    match = re.match(r'(\d)0(\d)0(\d)', str(exp_id))
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

df['a'] = df['实验编号'].apply(lambda x: parse_id(x)[0])
df['b'] = df['实验编号'].apply(lambda x: parse_id(x)[1])
df['c'] = df['实验编号'].apply(lambda x: parse_id(x)[2])

print("有效行数:", len(df.dropna(subset=['a','b','c'])))
print("a 的取值:", sorted(df['a'].dropna().unique()))
print("b 的取值:", sorted(df['b'].dropna().unique()))
print("c 的取值:", sorted(df['c'].dropna().unique()))
print("前5个解析结果:")
print(df[['实验编号','a','b','c']].head())
