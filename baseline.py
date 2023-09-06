from torch.nn import ModuleList, Conv2d, LazyLinear, Softmax, Module, ReLU, Flatten
from torch import save as tsave


class BaselineGenreClassifier(Module):
    def __init__(self, num_conv_layers=3, num_conv_filters=16, kernel_size=7, stride=3, num_dense_layers=2, num_dense_units=128, num_classes=10):
        super().__init__()
        ''' Implements a very basic CNN with no pooling(because pooling implies local translational invariance and that isn't the case in a mel spectrogram)'''
        self.version = 0
        self.num_conv_layers = num_conv_layers
        self.num_conv_filters = num_conv_filters
        self.num_dense_layers = num_dense_layers
        self.num_dense_units = num_dense_units
        self.num_classes = num_classes

        self.conv_blocks = ModuleList()
        self.dense_blocks = ModuleList()
        self.classifier_head = ModuleList()

        for i in range(self.num_conv_layers):
            if i==0:
                self.conv_blocks.append(Conv2d(3, self.num_conv_filters, kernel_size=kernel_size, stride=stride, padding='valid'))
            else:
                self.conv_blocks.append(Conv2d(self.num_conv_filters, self.num_conv_filters, kernel_size=kernel_size, stride=stride, padding='valid'))
            self.conv_blocks.append(ReLU())

        self.flatten = Flatten()

        for i in range(self.num_dense_layers):
            if i==0:
                self.dense_blocks.append(LazyLinear(self.num_dense_units))
            else:
                self.dense_blocks.append(LazyLinear(self.num_dense_units))
            self.dense_blocks.append(ReLU())

        self.classifier_head.append(LazyLinear(self.num_classes))
        self.classifier_head.append(Softmax(dim=1))

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.flatten(x)

        for block in self.dense_blocks:
            x = block(x)

        for layer in self.classifier_head:
            x = layer(x)

        return x
    
    def save(self, path):
        save_path = path + f'{self._get_name()}_{self.version}.pth'
        tsave(self.state_dict(), save_path)