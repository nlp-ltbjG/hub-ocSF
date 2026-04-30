import torch.nn as nn
import torch
import numpy as np


class Classifier(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.liner1=nn.Linear(input_size,hidden_size)
        self.liner2=nn.Linear(hidden_size,output_size)
        self.activate=nn.ReLU()

    def forward(self,x):
        x=self.liner1(x)
        x=self.activate(x)
        x=self.liner2(x)
        return x

model=Classifier(input_size=10,hidden_size=32,output_size=10)
#准备数据
def build_dataset(train_samples):
    rng = np.random.default_rng(seed=41)
    array=rng.standard_normal(size=(train_samples,10),dtype="float32")
    label=np.argmax(array,axis=1)
    return torch.from_numpy(array), torch.from_numpy(label)


#参数
lr=0.001#学习率
train_samples=1000#样本总数
batch_size=20#
#优化器
optimizer=torch.optim.Adam(params=model.parameters(),lr=lr)
#损失函数
loss_function = nn.CrossEntropyLoss()

inputs,labels=build_dataset(train_samples)


def evaluate(model):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 100
    x, y = build_dataset(total)

    with torch.no_grad():
        outputs = model(x)

        predicts = torch.argmax(outputs, dim=1)

        correct = (predicts == y).sum().item()

    accuracy = correct / total
    print(f"--- 验证集准确率: {accuracy * 100:.2f}% ---")
    return accuracy


def main():
    #开始训练
    for epoch in range(300):
        model.train()
        loss_list=[]
        for i in range(train_samples//batch_size):
            input1=inputs[i*batch_size:(i+1)*batch_size]
            label=labels[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            outputs = model(input1)#前向传播获得预测值
            loss=loss_function(outputs,label)
            loss.backward()#反向传播，获得梯度
            optimizer.step()#更新权重
            loss_list.append(loss.item())
        print("第{}轮 loss{}".format(epoch,np.mean(loss_list)))

        if (epoch + 1) % 10 == 0:  # 每10轮验证一次
            evaluate(model)

    #保存模型
    torch.save(model.state_dict(), "model.bin")

def test():
    """验证模型"""
    model.load_state_dict(torch.load("model.bin"))
    # print(model.state_dict())
    #验证集
    inputs,labels=build_dataset(10)
    model.eval()
    with torch.no_grad():
        for i in range(10):
            #前向传播
            logits = model(inputs[i]).unsqueeze(0)
            #计算概率
            probs = torch.nn.functional.softmax(logits, dim=1)
            #找到概率最大的类别索引
            predict_label = torch.argmax(logits, dim=1).item()
            #类别的具体概率值
            max_prob = torch.max(probs).item()
            print(f"输入: {inputs[i].numpy()}")
            print(f"预测类别: {predict_label}, 实际类别: {labels[i].item()}, 概率值: {max_prob:.4f}")


if __name__=="__main__":
    main()

    # test()
