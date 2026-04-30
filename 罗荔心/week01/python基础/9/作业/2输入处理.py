def get_positive_int():
    while True:
        try:
            value = float(input('请输入一个正数：'))
            if value > 0:#如果是反向判断，则continue
                break
            else:
                print(value,"不是正数。")
        except ValueError:
            print("请输入一个有效的数字。")
    return value
result1 = get_positive_int()
print(result1,"是正数。")

def get_age():
    while True:
        try:
            value = int(input('请输入你的年龄：'))
            if (value > 0)&(value < 150):
                break
            else:
                print(value,"为异常值。")
        except ValueError:
            print("请输入一个有效的整数。")
        #以下为额外的异常情况（根据参考答案补充）
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            return None
        except Exception as e:
            print(f"未知错误：{type(e).__name__}: {e}")
    return value
result2 = get_age()
print(result2,"是合理的年龄。")

def get_score():
    while True:
        try:
            value = float(input('请输入你的分数：'))
            if (value >= 0)&(value < 60):
                print("不及格。")
                break
            elif (value >= 60)&(value <=100):
                print("及格。")
                break
            else:
                print(value,"为异常值。")
        except ValueError:
            print("请输入一个有效的数字。")
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            return None
        except Exception as e:
            print(f"未知错误：{type(e).__name__}: {e}")
result3 = get_score()