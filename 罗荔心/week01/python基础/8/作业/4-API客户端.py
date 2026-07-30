import requests

base_url = 'https://jsonplaceholder.typicode.com'

def menu():
    print('1. 获取用户列表')
    print('2. 获取指定用户信息')
    print('3. 获取所有帖子')
    print('4. 获取指定用户的帖子')
    print('5. 退出')

while True:
    menu()
    choice = input('请输入功能选项：')
    
    if choice == '1':
        try:
            response = requests.get(base_url + '/users')
            response.raise_for_status()
            users = response.json()
            print(users) 
        except requests.exceptions.RequestException as e:
            print(f"请求失败：{e}")
        except ValueError as e:
            print(f"数据解析失败：{e}")
    
    elif choice == '2':
        user_id = input('请输入用户ID：')
        try:
            response = requests.get(base_url + '/users/' + user_id)
            response.raise_for_status()
            user = response.json()
            print(f"用户信息（ID: {user_id}）")
            print(f"姓名：{user['name']}")
            print(f"电话：{user['phone']}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"用户 {user_id} 不存在")
            else:
                print(f"HTTP错误：{e}")
        except requests.exceptions.RequestException as e:
            print(f"网络错误：{e}")
        except ValueError:
            print("返回数据格式错误")
    
    elif choice == '3':
        try:
            response = requests.get(base_url + '/posts')
            response.raise_for_status()
            posts = response.json()
            display_count = min(4, len(posts))#若总数少于设定值，只显示现有量
            print(f"显示前{display_count}条帖子")
            for post in posts[:4]:
                print(f"用户ID：{post['userId']}")
                print(f"标题：{post['title']}")
                print(f"内容：{post['body']}")
        except requests.exceptions.RequestException as e:
            print(f"请求失败：{e}")
    
    elif choice == '4':
        user_id = input('请输入用户ID：')
        try:
            response = requests.get(base_url + '/posts', params={'userId': user_id})
            response.raise_for_status()
            posts = response.json()
            print(f"用户ID {user_id} 的帖子：")
            if len(posts) == 0:
                print("该用户没有帖子")
            else:
                for post in posts:
                    print(f"帖子ID: {post['id']}")
                    print(f"标题: {post['title']}")
                    print(f"内容: {post['body']}")
        except requests.exceptions.RequestException as e:
            print(f"请求失败：{e}")
    
    elif choice == '5':
        break
    
    else:
        print("无效输入，请重新选择")