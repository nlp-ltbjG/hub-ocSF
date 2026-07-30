"""尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。"""
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch as tc
import torch.nn as nn
import numpy as np
# 创建单个训练数据
def create_data():
    x = np.random.random(4)
    #返回4维向量x和最大值下标
    maxindex = np.argmax(x)
    return x, maxindex
# 创建测试数据
def create_testdata(total):
    listdata = []
    for i in range(total):
        x = np.random.random(4)
        listdata.append(x)
    return listdata
#批量创建训练数据
def build_datas(total):
    xdata = []
    ydata = []
    for i in range(total):
        x,y = create_data()
        ydata.append(y)
        xdata.append(x)
    return tc.FloatTensor(np.array(xdata)), tc.LongTensor(np.array(ydata))

# 定义模型
class HelloModel(nn.Module):
    def __init__(self,input_size):
        super(HelloModel, self).__init__()
        self.linear = nn.Linear(input_size, 4)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 输出4维logits
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred
#开始训练
def starttrain():
    xl_luns = 20  # 训练轮数
    xl_batch = 100  # 每次训练样本个数
    xl_totalyb = 500000  # 总共训练的样本总数
    input_size = 4  # 输入向量维度
    learing_rate = 0.1 # 学习率
    #建立模型
    model = HelloModel(input_size)
    # 选择优化器
    optim = tc.optim.Adam(model.parameters(), lr=learing_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_datas(xl_totalyb)
    for epoch in range(xl_luns):
        model.train()
        watch_loss = []
        for index in range(xl_totalyb // xl_batch):
            # 取出一批训练值和真实值
            x = train_x[index * xl_batch : (index + 1) * xl_batch]
            y = train_y[index * xl_batch : (index + 1) * xl_batch]
            #计算损失函数
            loss = model(x, y)
            #求梯度
            loss.backward()
            #更新权重
            optim.step()
            #梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    tc.save(model.state_dict(), 'hello.bin')

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 4
    model = HelloModel(input_size)
    model.load_state_dict(tc.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    success = 0  # 成功预测的样本个数
    fail = 0  # 预测失败的样本个数
    with tc.no_grad():  # 不计算梯度
        result = model.forward(tc.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        yclb = int(tc.argmax(res)) #预测到的类别
        sjlb = int(tc.argmax(tc.FloatTensor(vec))) #真实类别
        if yclb == sjlb:
            success += 1
        else:
           fail += 1
    print(f"一共预测{len(input_vec)}个样本，成功预测{success}个，预测失败{fail}，成功率为{success/(success+fail)}")  # 打印结果


def main():
    starttrain()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # test_vec = [[0.18889086,0.25229675,0.31082123,0.43504317],
    #             [0.54963533,0.6524256,0.75758807,0.85520434],
    #             [0.90797868,0.17482528,0.23625847,0.34675372],
    #             [0.49349776,0.59416669,0.62579291,0.71567412]]
    test_vec = create_testdata(1000000)
    predict("hello.bin", test_vec)

if __name__ == "__main__":
    main()

