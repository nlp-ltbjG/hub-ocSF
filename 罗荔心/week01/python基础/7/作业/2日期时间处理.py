import datetime

now = datetime.datetime.now()
print(f'现在的时间是：{now}')
print(f'此时此刻是：{now.strftime("%Y年%m月%d日 %A %H:%M:%S")}')#带格式

today = now.date()
birthday = datetime.date(2001, 2, 11)
days = (today - birthday).days#带上days才能去掉具体时间，只保留天数
print(f'我的生日在{birthday},已经出生：{days}天了')
print(f'我的百日宴是在：{birthday + datetime.timedelta(days=100)}')
#timedelta是指时间上的偏移量