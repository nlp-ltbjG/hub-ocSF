import requests

#timeout是为了避免网络请求超时，程序无限等待
#get-200
url = 'https://jsonplaceholder.typicode.com/users'
try:
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        users = response.json()
        print(f'获取到{len(users)}个用户信息')
        print('前3个用户信息如下：')
        for user in users[:3]:
            print(f'用户名：{user["username"]} - 邮箱：{user["email"]}')

#post-201
    data = {'username': 'tom','age': 25, 'gender': 'male'}
    response = requests.post(url, json = data, timeout=5)
    if response.status_code == 201:
        result = response.json()
        print('用户信息添加成功')
        print(f"服务器返回数据：{result}")
    else:
        print('用户信息添加失败')
        print(response.status_code)

except requests.exceptions.RequestException as e:
    print(f"请求错误：{e}")
except Exception as e:
    print(f"发生错误：{e}")