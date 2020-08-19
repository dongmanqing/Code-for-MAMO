from utils import *

# ======================Embedding=========================
# item embedding
class ItemEmbedding(torch.nn.Module):
    def __init__(self, n_layer, in_dim, embedding_dim, activation='sigmoid'):
        super(ItemEmbedding, self).__init__()
        self.input_size = in_dim

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size/2)

        for i in range(n_layer - 1):
            linear_model = torch.nn.Linear(last_size, hid_dim)
            linear_model.bias.data.fill_(0.0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out

# user embedding
class UserEmbedding(torch.nn.Module):
    def __init__(self, n_layer, in_dim, embedding_dim, activation='sigmoid'):
        super(UserEmbedding, self).__init__()
        self.input_size = in_dim

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size / 2)

        for i in range(n_layer - 1):
            linear_model = torch.nn.Linear(last_size, hid_dim)
            linear_model.bias.data.fill_(0.0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out
