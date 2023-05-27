import torch
import torch.nn as nn
import torch.optim as optim

class State2EventNet(nn.Module):
    def __init__(self, input_vec_size, output_vec_size, no_hidden_layers, neurons_per_layer):
        super(State2EventNet, self).__init__()
        self.layer_seq = nn.Sequential(*self._get_modules(input_vec_size, output_vec_size, no_hidden_layers, neurons_per_layer))

    def _get_modules(self, input_size, output_size, num_hidden_layers, num_neurons):
        modules = None
        if num_hidden_layers == 0:
            modules = [nn.Linear(input_size, output_size)]
        else:
            modules = [nn.Linear(input_size, num_neurons), nn.ReLU()]
            for _ in range(num_hidden_layers):
                modules.append(nn.Linear(num_neurons, num_neurons))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(num_neurons, output_size))
        return modules

    def forward(self, input):
        return self.layer_seq(input)