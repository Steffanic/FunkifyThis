from torch.nn import ModuleList, Conv2d, LazyLinear, Softmax, Module, ReLU, Flatten

from torch import save as tsave

class FreqConvOnlyGenreClassifier(Module):
    def __init__(self, num_conv_layers=7, num_conv_filters=32, kernel_size=3, stride=2, num_dense_layers=2, num_dense_units=128, num_classes=10):
        super().__init__()
        ''' Implements a CNN with no pooling(because pooling implies local translational invariance and that isn't the case in a mel spectrogram) that only convolutional layers that operate on the frequency axis'''
        self.version=3
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
                self.conv_blocks.append(Conv2d(3, self.num_conv_filters, kernel_size=(kernel_size,1), stride=(stride, 1), padding='valid'))
            else:
                self.conv_blocks.append(Conv2d(self.num_conv_filters, self.num_conv_filters,  kernel_size=(kernel_size,1), stride=(stride, 1), padding='valid'))
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
        # instead of flattening we will swap the axes to be Batch x Time x Channel
        # that didn't seem to work tbh
        #x = x.permute(0, 3,1,2)
        # now we can flatten the time and channel axes
        #x = x.flatten(start_dim=2)

        for block in self.dense_blocks:
            x = block(x)

        for layer in self.classifier_head:
            x = layer(x)
        
        # now we have a softmax for each time bucket
        # we will average the softmaxes across the time axis
        #x = x.mean(dim=1)   

        return x
    
    def save(self, path):
        save_path = path + f'{self._get_name()}_{self.version}.pth'
        tsave(self.state_dict(), save_path)