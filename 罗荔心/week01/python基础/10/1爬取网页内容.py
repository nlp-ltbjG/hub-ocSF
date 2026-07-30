import requests
import bs4

def get(url):
        # 1.设置请求头，请求网页，并打印状态码
        try:
            headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(url,headers=headers, timeout=10)
            print(f'请求：{url}')
            print(f'响应状态码{response.status_code}')

            if response.status_code == 200:
                response.encoding = 'utf-8'
                soup = bs4.BeautifulSoup(response.text, 'html.parser')
                #title = soup.title.string 返回的是字符串或报错，含有子标签时会返回none
                #title = soup.title返回的是第一个匹配项
# 2.获取网页标题,打印网页的前500个字符
                title = soup.find('title')#可以通过class指定参数，返回所有匹配项
                title_text = title.text if title else "未找到标题"#获取的是纯文本
                print(f'网页标题：{title_text}')

                print(f'网页前500个字符：{response.text[:500]}')
                return response.text

# 3.使用异常处理 处理网页错误
        except requests.exceptions.Timeout:
            print("错误：请求超时！")
            return None
        except requests.exceptions.ConnectionError:
            print("错误：连接失败，请检查网络连接！")
            return None
        except requests.exceptions.RequestException as e:
            print(f'请求错误：{type(e).__name__}: {e}')
            return None
        except Exception as e:
            print(f'未知错误：{type(e).__name__}: {e}')
            return None

url = 'https://www.northnews.cn/'

if get(url):
    print('成功获取网页内容')
else:
    print('获取网页内容失败')