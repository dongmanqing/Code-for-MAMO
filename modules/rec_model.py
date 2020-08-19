from utils import *


class RecMAM(torch.nn.Module):
    def __init__(self, embedding_dim, n_y, n_layer, activation='sigmoid', classification=True):
        super(RecMAM, self).__init__()
        self.input_size = embedding_dim * 2

        self.mem_layer = torch.nn.Linear(self.input_size, self.input_size)

        fcs = []
        last_size = self.input_size

        for i in range(n_layer - 1):
            out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, out_dim)
            fcs.append(linear_model)
            last_size = out_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        if classification:
            finals = [torch.nn.Linear(last_size, n_y), activation_func('softmax')]
        else:
            finals = [torch.nn.Linear(last_size, 1)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        out0 = self.mem_layer(x)
        out = self.fc(out0)
        out = self.final_layer(out)
        return out
