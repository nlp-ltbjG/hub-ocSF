from cmath import inf

import torch.nn as nn
import torch

import numpy as np
from transformers import BertModel

bert=BertModel.from_pretrained(r"D:\badouai\week4语言模型\bert-base-chinese\bert-base-chinese",
                               local_files_only=True,
                               trust_remote_code=True,
                               )
bert.eval()
state_dict=bert.state_dict()
# print(bert.config)

def softmax(x,axis=-1):
    x=x-np.max(x,axis=axis,keepdims=True)
    return np.exp(x)/np.sum(np.exp(x),axis=axis,keepdims=True)

def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))


class MyBert:
    def __init__(self, state_dict):
        self.num_attention_heads=bert.config.num_attention_heads
        self.hidden_size=bert.config.hidden_size
        self.num_hidden_layers=bert.config.num_hidden_layers
        self.load_weights(state_dict)
    def load_weights(self,state_dict):
        self.word_embeddings=state_dict['embeddings.word_embeddings.weight'].numpy()
        self.position_embeddings=state_dict['embeddings.position_embeddings.weight'].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        for i in range(self.num_attention_heads):
            q_w = state_dict[f"encoder.layer.{i}.attention.self.query.weight"].numpy()
            q_b = state_dict[f"encoder.layer.{i}.attention.self.query.bias"].numpy()
            k_w = state_dict[f"encoder.layer.{i}.attention.self.key.weight"].numpy()
            k_b = state_dict[f"encoder.layer.{i}.attention.self.key.bias"].numpy()
            v_w = state_dict[f"encoder.layer.{i}.attention.self.value.weight"].numpy()
            v_b = state_dict[f"encoder.layer.{i}.attention.self.value.bias"].numpy()
            attention_output_weight = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"].numpy()
            attention_output_bias = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"].numpy()
            attention_layer_norm_w = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy()
            attention_layer_norm_b = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy()
            intermediate_weight = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"].numpy()
            intermediate_bias = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
            output_weight = state_dict[f"encoder.layer.{i}.output.dense.weight"].numpy()
            output_bias = state_dict[f"encoder.layer.{i}.output.dense.bias"].numpy()
            ff_layer_norm_w = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
            ff_layer_norm_b = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                 output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
            #pooler
            self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
            self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    def embedding_forward(self, x):
        #x[seq_length]
        word_embedding = self.get_embedding(self.word_embeddings, x)
        #需要位置编号[0,1,2,3]
        position_embedding = self.get_embedding(self.position_embeddings, list(range(len(x))))
        #句子类型id
        if 102 in x:
            first_sep_index = x.index(102)
        else:
            first_sep_index = len(x) - 1
        token_type_id = []
        for i in range(len(x)):
            if i <= first_sep_index:
                token_type_id.append(0)
            else:
                token_type_id.append(1)
        token_type_embedding = self.get_embedding(self.token_type_embeddings, token_type_id)
        embedding=word_embedding+position_embedding+token_type_embedding
        embedding=self.layer_norm(embedding,self.embeddings_layer_norm_weight,self.embeddings_layer_norm_bias)
        return embedding

    def get_embedding(self, embeddings, x):
        """
        获取embedding,这里只展示了查表法获取，实际也可利用one-hot@embedding矩阵，但是内存消耗大，实际工程通常直接查表
        :param embeddings: 矩阵embeddings[vocal_size,hidden_size]
        :param x: x是[101,125,456,789]这样的token_id序列
        :return: 返回embedding[seq_length,hidden_size]
        """
        return np.array([embeddings[index] for index in x])
    def all_self_attention(self,x):
        for i in range(self.num_hidden_layers):
            x=self.single_transformer_layer_forward(x,i)
        return x

    def single_transformer_layer_forward(self,x,layer_index):
        weights = self.transformer_weights[layer_index]
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        attention_output=self.self_attention(x,q_w,q_b,k_w,k_b,v_w,v_b,attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #LN（残差连接）
        x=self.layer_norm(x+attention_output, attention_layer_norm_w, attention_layer_norm_b)

        #FFN
        feed_forward_x=self.feed_forward(x,intermediate_weight, intermediate_bias, output_weight, output_bias)

        #LN(残差连接)
        x=self.layer_norm(x+feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x
    def self_attention(self,x,q_w,q_b,k_w,k_b,v_w,v_b,attention_output_weight, attention_output_bias,
                                num_attention_heads,
                                hidden_size):
        """
        x[seq_length,hidden_size]
        q_w[hidden_size,hidden_size]
        """
        q=np.dot(x,q_w.T)+q_b  #[seq_length,hidden_size]
        k=np.dot(x,k_w.T)+k_b  #[seq_length,hidden_size]
        v=np.dot(x,v_w.T)+v_b  #[seq_length,hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        q=self.mulit_head(q,num_attention_heads,attention_head_size) #[num_attention_heads, seq_length, attention_head_size]
        k=self.mulit_head(k,num_attention_heads,attention_head_size) #[num_attention_heads, seq_length, attention_head_size]
        v=self.mulit_head(v,num_attention_heads,attention_head_size) #[num_attention_heads, seq_length, attention_head_size]

        qk=q@k.swapaxes(1,2)
        qk/=np.sqrt(attention_head_size)
        qk=softmax(qk) #[num_attention_heads, seq_length, seq_length]
        qkv=np.matmul(qk,v) #[num_attention_heads, seq_length, attention_head_size]
        qkv=qkv.swapaxes(0,1).reshape(-1,hidden_size) #把12个头拼接起来[seq_length,hidden_size]
        attention=qkv@attention_output_weight.T+attention_output_bias# 这里的W需要转置，因为pytorch的权重形状是[out_features, in_features]
        return attention

    def feed_forward(self,x,intermediate_weight, intermediate_bias,output_weight, output_bias):
        x=x@intermediate_weight.T+intermediate_bias
        x=gelu(x)
        x=x@output_weight.T+output_bias
        return x

    def mulit_head(self,x,num_attention_heads,attention_head_size):
        max_len, hidden_size = x.shape #[seq_length,hidden_size]
        x = x.reshape(max_len, num_attention_heads, attention_head_size) #[seq_length,heads,attention_head_size]
        x = x.swapaxes(1, 0)  # 交换第1维和第0维 [num_attention_heads, seq_length, attention_head_size]
        return x

    def layer_norm(self,x,w,b):
        mean=x.mean(axis=-1,keepdims=True)#如果x是numpy 数组，自身有方法.mean,np.mean(x)时，x可以是普通列表，但不能是张量
        var=((x-mean)**2).mean(axis=-1,keepdims=True)
        x_norm=((x-mean)/np.sqrt(var+1e-12))*w+b
        return x_norm

    #对[cls]进行处理
    def pooler_output_layer(self,x):
        x=x@self.pooler_dense_weight.T+self.pooler_dense_bias
        x=np.tanh(x)
        return x

    def forward(self,x):
        x=self.embedding_forward(x)#[seq_length,hidden_size]
        sequence_output=self.all_self_attention(x)#[seq_length,hidden_size]
        pooler_output=self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


x=[741,456,789,386,457]
my_bert=MyBert(state_dict)
my_sequence_output,my_pooler_output=my_bert.forward(x)

print(my_sequence_output)
# print(my_pooler_output)

torch_x=torch.LongTensor([x])
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(torch_sequence_output)
# print(torch_pooler_output)