import torch
import torch.nn as nn

from utils.conv_block import ConvBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DoubleRelationNet(nn.Module):
    def __init__(self, n_way, k_support, k_query, k_query_val, conv_dim, fc_dim):
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

        self.init_params()

    def forward(self, support_vector, query_vector, is_infer=False):
        num_query = self.k_query if self.is_train else self.k_query_val

        _size = support_vector.size()

        support_vector = support_vector.view(self.n_way, self.k_support, _size[1], _size[2], _size[3]).mean(dim=1)
        if not is_infer:
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

        return out.view(-1, self.n_way)

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
