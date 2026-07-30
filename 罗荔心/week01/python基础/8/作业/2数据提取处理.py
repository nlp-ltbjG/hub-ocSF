import requests

url = 'https://jsonplaceholder.typicode.com/posts'

#统计帖子总数
response = requests.get(url, timeout=5)
if response.status_code == 200:
    posts = response.json()
    print(f'该网站共有{len(posts)}篇帖子')
else:
    print('请求失败')
    print(response.status_code)

#统计id=1的用户的所有帖子
response = requests.get('https://jsonplaceholder.typicode.com/posts?userId=1', timeout=5)
if response.status_code == 200:
    posts = response.json()
    print(f'该用户共有{len(posts)}篇帖子')
#显示该用户前五个帖子的标题
    for idx, post in enumerate(posts[:5], start=1):
        print(f'第{idx}篇帖子标题：{post['title']}')
#找出其中标题最长的帖子
#因为需要进行比较，所以要定义函数，或使用key=lambda x: len(x['title']）
def get_title_length(x):
    return len(x['title'])
max_title = max(posts, key=get_title_length)
print(f'标题最长的帖子是：{max_title["title"]},长度为{len(max_title["title"])}')