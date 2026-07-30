import pandas as pd
import os
import random

data = ['John', 'Anna', 'Peter', 'Linda', 'Tom', 'Alice', 'Bob', 'Lucy', 'Mike', 'Sara']
df = pd.DataFrame(data, columns=['姓名'])

script_dir = os.path.dirname(os.path.abspath(__file__))
# 指定 CSV 文件的完整路径
file_path = os.path.join(script_dir, '抽奖名单.csv')
# 保存
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print("已创建抽奖名单")

#处理文件不存在的情况
try:
    df_read = pd.read_csv(file_path, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"抽奖名单文件不存在：{file_path}")
    exit(1)

name_list = df_read['姓名'].tolist()
if len(name_list) < 3:
    print("抽奖人数少于抽奖次数，无需抽奖")
else:
    winners = random.sample(name_list, 3)
    print("中奖者是：", "、".join(winners))

result_file = os.path.join(script_dir, '中奖结果.csv')
df_result = pd.DataFrame(winners, columns=['中奖者'])
df_result.to_csv(result_file, index=False, encoding='utf-8-sig')
print(f"中奖结果已保存到：{result_file}")

#测试占用内存
import sys
print("dataframe大小：",sys.getsizeof(df_read))
print("name_list大小：",sys.getsizeof(name_list))