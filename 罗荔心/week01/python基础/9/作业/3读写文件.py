import os
import tempfile
import sys

def read_file_safe(file_name):
    try:
        #要限定读取文件的编码规则
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            print("文件读取成功")
            return content
    except FileNotFoundError:
        print("文件不存在") 
        return None
    except PermissionError:
        print("没有权限读取文件") 
        return None
    except UnicodeDecodeError:
        print("文件编码错误") #此处还可以再嵌套一层try except来处理编码错误，使用gbk等
        return None
   #直接使用return"内容"，实际没有打印错误信息，只是返回状态字符串，需print后再return none

def write_file_safe(file_name, content):
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)
            print("文件写入成功")
            return content
    except FileNotFoundError:
        print("文件不存在")
        return None
    except PermissionError:
        print("没有权限写入文件")
        return None
    except UnicodeEncodeError:
        print("文件编码错误")
        return None
    
def copy_file_safe(source_file, target_file):
    content = read_file_safe(source_file)
    if content is None:#不存在、无权限、编码错误等情况
        return False
    if not content:
        print("文件内容为空，无法复制")
        return False
    success = write_file_safe(target_file, content)
    if success:
        print("文件复制成功")
    else:
        print("文件复制失败")
    return success

#测试（使用临时目录）
def run_tests():
    # 创建临时目录，自动保证可写，测试结束后自动删除
    with tempfile.TemporaryDirectory() as tmpdir:
        # 切换到临时目录
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        print(f"测试目录（可写）: {tmpdir}")

# 1. 测试 read_file_safe
        print("\n1. 测试 read_file_safe 函数：")
        print("\n测试1：读取不存在的文件")
        content1 = read_file_safe("不存在的文件.txt")

        print("\n测试2：创建并读取文件")
        test_content = "测试文件"
        write_file_safe("test_read.txt", test_content)

        print("\n测试3：读取存在的文件")
        content2 = read_file_safe("test_read.txt")
        if content2:
            print("文件内容：")
            print(content2)

        # 2. 测试 write_file_safe
        print("\n2. 测试 write_file_safe 函数：")
        print("\n测试1：写入新文件")
        content3 = "新创建的文件"
        success1 = write_file_safe("test_write.txt", content3)

        print("\n测试2：覆盖已有文件")
        content4 = "更新后的内容"
        success2 = write_file_safe("test_write.txt", content4)

        # 3. 测试 copy_file_safe
        print("\n3. 测试 copy_file_safe 函数：")
        print("\n测试1：复制文件")
        success3 = copy_file_safe("test_read.txt", "test_copy.txt")

        print("\n测试2：复制不存在的文件")
        success4 = copy_file_safe("不存在的源文件.txt", "target.txt")

        # 4. 验证复制结果
        print("\n4. 验证复制结果：")
        if os.path.exists("test_copy.txt"):
            content5 = read_file_safe("test_copy.txt")
            if content5:
                print("复制后的文件内容：")
                print(content5)

 # 清理临时文件（TemporaryDirectory 会自动清理，但也可以手动确认）
        print("\n测试完成，临时目录将被自动删除。")
        # 切回原目录
        os.chdir(original_dir)

if __name__ == "__main__":
    run_tests()