import os
import json

class StudentScoreManager:
    def __init__(self, filename="student_scores.json"):
        self.filename = filename
        self.scores = {}      # 格式: {"张三": [85, 90, 88], "李四": [92]}
        self.load_scores()

    def load_scores(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.scores = json.load(f)
            print(f"成功加载分数数据，共计 {len(self.scores)} 个学生。")
        except FileNotFoundError:
            print("文件不存在，将创建新文件。")
            self.scores = {}#文件不存在或损坏时初始化为空字典
        except json.JSONDecodeError:
            print("JSON 文件损坏，将创建新文件。")
            self.scores = {}
        except PermissionError:
            print("没有权限读取文件，将创建新文件。")
            self.scores = {}

    def save_scores(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.scores, f, ensure_ascii=False, indent=4)
            return True
        except PermissionError:
            print(f"错误：没有权限写入文件 {self.filename}！")
            return False
        except OSError as e:
            print(f"错误：保存文件失败：{type(e).__name__}: {e}")
            return False
        except Exception as e:
            print(f"保存成绩数据时出错：{type(e).__name__}: {e}")
            return False

    def add_score(self, name, score):
        """
        添加学生成绩（支持多次添加）
        返回 True 表示成功，False 表示失败
        """
        # 1. 校验分数有效性
        if not isinstance(score, (int, float)):
            print("错误：分数必须是数字！")
            return False
        if score < 0 or score > 100:
            print("错误：分数必须在 0~100 之间！")
            return False

        # 2. 添加成绩
        if name in self.scores:
            self.scores[name].append(score)
            print(f"已为 {name} 添加成绩：{score} 分")
        else:
            self.scores[name] = [score]
            print(f"已为新学生 {name} 添加成绩：{score} 分")

        # 3. 保存到文件
        if self.save_scores():
            print("成绩已保存")
            return True
        else:
            print("成绩保存失败")
            return False

    def get_average_score(self, name):
        if name not in self.scores:#返回指定学生的平均分
            print(f"错误：学生 {name} 不存在")
            return None
        scores = self.scores[name]
        if not scores:
            print(f"错误：学生 {name} 没有成绩记录")
            return None
        avg = sum(scores) / len(scores)
        return round(avg, 2)#保留两位小数

    def display_scores(self):
        """显示所有学生的所有成绩"""
        if not self.scores:
            print("暂无成绩数据")
            return
        for name, score_list in self.scores.items():
            # 将列表转为字符串显示，如 [85, 90, 88]
            scores_str = ", ".join(str(s) for s in score_list)
            print(f"{name}: {scores_str}")
        print("显示分数成功。")

#测试代码
#直接运行py文件时条件成立，执行缩进的代码块
if __name__ == "__main__":
    manager = StudentScoreManager("student_scores.json")

    print("\n1. 添加正常成绩：")
    manager.add_score("张三", 85)
    manager.add_score("张三", 90)
    manager.add_score("张三", 88)
    manager.add_score("李四", 92)
    manager.add_score("李四", 88)
    manager.add_score("王五", 78)

    print("\n2. 测试异常情况：")
    print("\n测试1：分数超出范围")
    manager.add_score("李四", 150)
    print("\n测试2：分数为负数")
    manager.add_score("王五", -10)
    print("\n测试3：分数不是数字")
    manager.add_score("王五", "abc")
    print("\n测试4：分数为空字符串")
    manager.add_score("王五", "")

    print("\n3. 查询平均分：")
    avg1 = manager.get_average_score("张三")
    if avg1 is not None:
        print(f"张三的平均分：{avg1:.2f} 分")
    avg2 = manager.get_average_score("李四")
    if avg2 is not None:
        print(f"李四的平均分：{avg2:.2f} 分")

    print("\n测试：查询不存在的学生")
    avg3 = manager.get_average_score("赵六")

    print("\n4. 显示所有成绩：")
    manager.display_scores()