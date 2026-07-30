import pandas as pd
import os

data = {
    '名字': ['John', 'Anna', 'Peter', 'Linda'],
    '分数': [78, 54, 65, 82],
    '出生地': ['New York', 'Paris', 'Berlin', 'London']
}

df = pd.DataFrame(data)
print("创建的DataFrame:")
print(df)

# 获取当前 .py 文件所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 指定 CSV 文件的完整路径
file_path = os.path.join(script_dir, '学生信息.csv')
# 保存
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print("已同步创建学生信息文件")

# a为追加写入，不覆盖原有数据
newdata = {
    'Tom,91,Madrid',
    'Lily,88,Beijing'
}
with open(file_path, 'a', encoding='utf-8') as f:
    for record in newdata:
        f.write(record + '\n')

    print(f'已追加以下数据：')
    for record in newdata:
        print(record)

# 重新打开，使用 readlines()
with open(file_path, 'r', encoding='utf-8') as f:
    print(f'最新数据：' + '\n',f.read())