import torch
from torch import nn
import torch.nn.functional as F


class FeedforwardLayer(nn.Module):

    def __init__(self, d_model):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model*4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(F.leaky_relu(output, 0.1))
        output = self.dropout(output)
        return self.norm(input+output)


class DimensionBufferLayer(nn.Module):

    def __init__(self, d_in, d_out, d_hidden):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_hidden, d_out)

    def forward(self, input):
        output = self.linear1(input)
        output = self.dropout(output)
        output = self.linear2(F.leaky_relu(output, 0.1))
        return output


class SequenceEdgeMeanPooling(nn.Module):
    """
    concat first, last and mean from sequence
    input: (seq_len, *, dim)
    output: (*, dim*3)
    """

    def forward(self, input):
        return torch.cat([input[0], input.mean(dim=0), input[-1]], dim=-1)


class BERTStyleRNNBlock(nn.Module):

    def __init__(self, d_model):
        nn.Module.__init__(self)
        assert d_model % 2 == 0
        self.rnn = nn.LSTM(d_model, d_model//2, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.feed = FeedforwardLayer(d_model)

    def forward(self, seq_tns):
        out_tns, _ = self.rnn(seq_tns)
        out_tns = self.norm(out_tns+seq_tns)
        return self.feed(out_tns)


class RNNFeatureEncoder(nn.Module):
    """
    input: (batch, seq_len, d_feature)
    output: (batch, d_model)
    """

    def __init__(self, *, d_model, l_rnn, d_feature):
        nn.Module.__init__(self)
        self.projection = DimensionBufferLayer(d_feature, d_model, d_model*4)
        self.rnn = nn.Sequential(*[BERTStyleRNNBlock(d_model) for _ in range(l_rnn)])
        self.pooling = nn.Sequential(
            SequenceEdgeMeanPooling(),
            DimensionBufferLayer(d_model*3, d_model, d_model*4))

    def forward(self, input):
        tns = input.permute(1, 0, 2)  # (seq_len, batch, 8)
        tns = self.projection(tns)    # (seq_len, batch, d_model)
        tns = self.rnn(tns)
        tns = self.pooling(tns)       # (batch, d_model)
        return tns


# source: https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))

class GroupLinear(nn.Module):
    """
    input: (batch, d_in)
    output: (batch, groups, d_out)
    """

    def __init__(self, d_in, d_out, groups, align_coef=0.99):
        nn.Module.__init__(self)
        self.linears = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(groups)])
        source = self.linears[0]
        for i_group in range(1, groups):
            dest = self.linears[i_group]
            polyak_update(align_coef, dest, source)

    def forward(self, input):
        outputs = [func(input) for func in self.linears]  # (groups, batch, d_out)
        output = torch.stack(outputs, dim=1)  # (batch, groups, d_out)
        return output


class MSERegressionStabilizer(nn.Module):

    def __init__(self, mu, sigma):
        nn.Module.__init__(self)
        self.mu = mu
        self.sigma = sigma

    def forward(self, scalars):
        return scalars * self.sigma + self.mu

    def get_mse(self, y_pred, y_true):
        mse = (y_pred - y_true) ** 2
        mse /= self.sigma
        return torch.mean(mse)

