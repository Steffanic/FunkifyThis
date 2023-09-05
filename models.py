from torch.nn import ModuleList, Conv2d, LazyLinear, Softmax, Module, ReLU, Flatten
from torchinfo import summary

from infrastructure import get_gtzan_dataloader, train_to_classify_genres, plot_loss_accuracy

class BaselineGenreClassifier(Module):
    def __init__(self, num_conv_layers=3, num_conv_filters=16, kernel_size=7, stride=3, num_dense_layers=2, num_dense_units=128, num_classes=10):
        super().__init__()
        ''' Implements a very basic CNN with no pooling(because pooling implies local translational invariance and that isn't the case in a mel spectrogram)'''
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

if __name__=='__main__':
    gtzan_dataloader, genre_from_class_id = get_gtzan_dataloader(image_folder_root="D:\GTZAN\Data\images_original", batch_size=64, shuffle=True)
    model = BaselineGenreClassifier()
    # grab the first batch for a summary
    mel_spectrogram, genre = next(iter(gtzan_dataloader))
    summary(model, input_size=mel_spectrogram.shape)
    loss_history_dict, accuracy_history_dict = {}, {}

    loss_history, accuracy_history = train_to_classify_genres(model, gtzan_dataloader, num_epochs=10, learning_rate=0.001, learning_rate_gamma=0.9, model_save_path='model_saves/baseline_genre_classifier.pth')

    print('Done.')

    loss_history_dict['BaselineGenreClassifier'] = loss_history
    accuracy_history_dict['BaselineGenreClassifier'] = accuracy_history
    plot_loss_accuracy(loss_history_dict, accuracy_history_dict, ['BaselineGenreClassifier'], plot_save_path='plots/baseline_genre_classifier.png')
