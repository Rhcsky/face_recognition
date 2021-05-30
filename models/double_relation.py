import torch
import torch.nn as nn

from utils.common import split_support_query_set
from utils.conv_block import ConvBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Embedding(nn.Module):
    """
    Model as described in the reference paper,
    """

    def __init__(self, in_channel=3, z_dim=64):
        super().__init__()
        self.block1 = ConvBlock(in_channel, z_dim, 3, max_pool=2)
        self.block2 = ConvBlock(z_dim, z_dim, 3, max_pool=2)
        self.block3 = ConvBlock(z_dim, z_dim, 3, max_pool=None, padding=1)
        self.block4 = ConvBlock(z_dim, z_dim, 3, max_pool=None, padding=1)

        self.init_params()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class DoubleRelationNet(nn.Module):
    def __init__(self, n_way, k_support, k_query, k_query_val, in_channel, conv_dim, fc_dim):
        super().__init__()
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.k_query_val = k_query_val

        self.layer1 = ConvBlock(64 * 2, conv_dim, 3, max_pool=2)
        self.layer2 = ConvBlock(conv_dim, 64, 3, max_pool=2)
        self.fc1 = nn.Linear(960, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

        self.is_train = True
        self.embedding = Embedding(in_channel).to(device)

        self.init_params()

    def forward(self, x, y):
        embedding = self.embedding(x)

        num_query = self.k_query if self.is_train else self.k_query_val
        support_vector, query_vector, y_support, y_query = split_support_query_set(embedding, y, self.n_way,
                                                                                   self.k_support,
                                                                                   num_query)
        _size = support_vector.size()

        support_vector = support_vector.view(self.n_way, self.k_support, _size[1], _size[2], _size[3]).mean(dim=1)
        support_vector = support_vector.repeat(self.n_way * num_query, 1, 1, 1)
        query_vector = torch.stack([x for x in query_vector for _ in range(self.n_way)])

        bilinear = torch.mul(support_vector, query_vector)
        distance = torch.subtract(support_vector, query_vector).pow(2)

        out = torch.cat((bilinear, distance), dim=1)

        out = self.layer1(out)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out.view(-1, self.n_way), y_query

    def custom_train(self):
        self.is_train = True

    def custom_eval(self):
        self.is_train = False

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
