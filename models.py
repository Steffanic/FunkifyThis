from functools import partial
from multiprocessing import Pool
import time
from torch.nn import ModuleList, Conv2d, LazyLinear, Softmax, Module, ReLU, Flatten
from torchinfo import summary
from torch import save as tsave
from baseline import BaselineGenreClassifier
from frequencyOnlyModel import FreqConvOnlyGenreClassifier

from infrastructure import get_gtzan_dataloader, train_to_classify_genres, plot_loss_accuracy


if __name__=='__main__':
    train_gtzan_dataloader, val_gtzan_dataloader, genre_from_class_id = get_gtzan_dataloader(image_folder_root="D:\GTZAN\Data\images_original", batch_size=256, shuffle=True)
    models = [BaselineGenreClassifier(), FreqConvOnlyGenreClassifier()]
    # grab the first batch for a summary
    mel_spectrogram, genre = next(iter(val_gtzan_dataloader))
    for model in models:
        summary(model, input_size=mel_spectrogram.shape)
    loss_history_dict, accuracy_history_dict = {}, {}

    histories = map(partial(train_to_classify_genres, train_gtzan_dataloader=train_gtzan_dataloader, val_gtzan_dataloader=val_gtzan_dataloader, num_epochs=20, learning_rate=0.001, learning_rate_gamma=0.99, model_save_path='model_saves/'), models)


    print('Done.')


    for model, history in zip(models, histories):
        train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = history
        loss_history_dict[type(model).__name__+"_train"] = train_loss_history
        accuracy_history_dict[type(model).__name__+"_train"] = train_accuracy_history
        loss_history_dict[type(model).__name__+"_val"] = val_loss_history
        accuracy_history_dict[type(model).__name__+"_val"] = val_accuracy_history
    
    plot_loss_accuracy(loss_history_dict, accuracy_history_dict, model_names=[type(m).__name__ for m in models], plot_save_path=f'plots/{[type(m).__name__ for m in models]}_loss_accuracy.png')