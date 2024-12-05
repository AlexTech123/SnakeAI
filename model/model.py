import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.l2(out)

        return out

    def save(self, file_name='model.pth'):
        model_folder_path = './model'

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)