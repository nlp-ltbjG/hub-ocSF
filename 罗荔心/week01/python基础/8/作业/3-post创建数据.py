import requests

url = 'https://jsonplaceholder.typicode.com/posts'
new_post = {'title': '帖子', 'body': '内容', 'userId': 1}
response = requests.post(url, json = new_post, timeout=5)
if response.status_code == 201:
    result = response.json()
    print(f'返回数据：{result}')
    print('创建成功,状态码：', response.status_code)
else:
    print('创建失败,状态码：', response.status_code)