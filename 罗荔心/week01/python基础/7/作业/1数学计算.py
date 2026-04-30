import math
import random

r = 3
print(f'半径为{r}的圆，面积={math.pi * r ** 2}')
print(f'半径为{r}的圆，周长={2 * math.pi * r}')

num = 9
print(f'{num}的平方根为{math.sqrt(num)}')

a = 10
b = 100
print(f'以{a}底求{b}的对数={math.log(b, a)}')

print('在1到100的闭区间中随机数取一个随机整数',random.randint(1, 100))
print('在1到100的闭区间中随机数取一个随机浮点数',random.uniform(1, 100))
print('在1到100的左闭右开区间中随机数取一个随机整数',random.randrange(1, 100))
print('在0到1的左闭右开区间中随机数取一个随机浮点数',random.random())

my_list = [1, 2, 3, 4, 5]
print(random.choice(my_list))