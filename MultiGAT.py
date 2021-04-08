import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F


class GAL(MessagePassing):
    def __init__(self,in_features,out_featrues):
        super(GAL,self).__init__(aggr='add')
        self.a = torch.nn.Parameter(torch.zeros(size=(2*out_featrues, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        # 定义leakyrelu激活函数
        self.leakyrelu = torch.nn.LeakyReLU()
        self.linear=torch.nn.Linear(in_features,out_featrues)
    def forward(self,x,edge_index):
        x=self.linear(x)
        N=x.size()[0]
        edge_index = torch.where(edge_index>0 , 1, 0)
        row,col=edge_index
        a_input = torch.cat([x[row], x[col]], dim=1)
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        temp=torch.mm(a_input,self.a).squeeze()
        e = self.leakyrelu(temp)
        #e_all为同一个节点与其全部邻居的计算的分数的和，用于计算归一化softmax
        e_all=torch.zeros(x.size()[0])
        count = 0
        for i in col:
            e_all[i]+=e[count]
            count=count+1

        for i in range(len(e)):
            e[i]=math.exp(e[i])/math.exp(e_all[col[i]])

        return self.propagate(edge_index,x=x,norm=e)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GAT(torch.nn.Module):
    def __init__(self, in_features, hid_features, out_features, n_heads):
        """
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()

        # 定义multi-head的图注意力层
        self.attentions = [GAL(in_features, hid_features) for _ in
                           range(n_heads)]
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GAL(hid_features * n_heads, out_features)

    def forward(self, x, edge_index):
        # 将每个head得到的x特征进行拼接
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        print('x.size after cat',x.size())
        x = F.elu(self.out_att(x,edge_index))  # 输出并激活
        print('x.size after elu',x.size())
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定

    def message(self, x_i, x_j, size_i, edge_index_i):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H x (emb(out)+ emb(out))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, edge_index_i, size_i)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)  # (E+N) x H x emb(out)






