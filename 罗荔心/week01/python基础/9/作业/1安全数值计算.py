def safe_divide(a, b):
    while True:
        try:
            c = a / b
            return c
        except ZeroDivisionError:
            print("除数不能为0，请重新输入")
            return None
        except TypeError:
            print("输入参数不是数字，请重新输入")
            return None
#result1 = safe_divide()
#print("结果是：", result1)

def safe_power(a, b):
    while True:
        try:
            c = a ** b
            return c
        except TypeError:
            print("输入参数不是数字，请重新输入")
            return None
#result2 = safe_power()
#print("结果是：", result2)

#测试
print("\n1. 测试safe_divide函数：")
# 正常情况
result1 = safe_divide(10, 2)
print(f"10 / 2 = {result1}")

result2 = safe_divide(15, 3)
print(f"15 / 3 = {result2}")

# 除数为0
result3 = safe_divide(10, 0)
print(f"10 / 0 = {result3}")

# 类型错误
result4 = safe_divide(10, "2")
print(f"10 / '2' = {result4}")

result5 = safe_divide("10", 2)
print(f"'10' / 2 = {result5}")

print("\n2. 测试safe_power函数：")
# 正常情况
result6 = safe_power(2, 3)
print(f"2 ^ 3 = {result6}")

result7 = safe_power(5, 2)
print(f"5 ^ 2 = {result7}")

result8 = safe_power(16, 0.5)  # 开平方根
print(f"16 ^ 0.5 = {result8}")

# 类型错误
result9 = safe_power("2", 3)
print(f"'2' ^ 3 = {result9}")

result10 = safe_power(2, "3")
print(f"2 ^ '3' = {result10}")#若通过float进行转换，则不会报错