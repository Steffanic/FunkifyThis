from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

from torch import argmax as targmax, sum as tsum, save as tsave, Generator

from tqdm import tqdm 

import matplotlib.pyplot as plt

from time import time

def plot_loss_accuracy(loss_history, accuracy_history, model_names, plot_save_path=None):
    """
    Plots the loss and accuracy history for every model.
    
    Args:
    loss_history: A dict of lists, keys are the model_names and values are the loss histories.
    accuracy_history: A dict of lists, keys are the model_names and values are the accuracy histories.
    model_names: a list of model names to iterate through and plot histories for
    """
    plt.figure(figsize=(20, 10))
    for model_name in model_names:
        plt.subplot(1, 2, 1)
        plt.plot(loss_history[model_name+"_train"], label=f'{model_name} train loss')
        plt.plot(loss_history[model_name+"_val"], label=f'{model_name} val loss')
        plt.xlabel('Epoch*Batch')
        plt.ylabel('Loss')
        plt.title('Loss history')
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_history[model_name+"_train"], label=f'{model_name} train accuracy')
        plt.plot(accuracy_history[model_name+"_val"], label=f'{model_name} val accuracy')
        plt.xlabel('Epoch*Batch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy history')
    plt.legend()
    if plot_save_path is not None:
        plot_save_path = plot_save_path[:-4] + f'_{int(time())}.png'
        plt.savefig(plot_save_path)
    plt.show()


def get_gtzan_dataloader(image_folder_root, batch_size=64, shuffle=True):
    """Returns a dataloader for the GTZAN dataset."""   
    genre_image_dataset = ImageFolder(image_folder_root, transform=ToTensor())
    genre_from_class_id = lambda idx:[key for key in genre_image_dataset.class_to_idx.keys() if genre_image_dataset.class_to_idx[key]==idx][0]
    val_len = int(len(genre_image_dataset)*0.2)
    train_len = len(genre_image_dataset) - val_len
    lengths = [train_len, val_len]
    train_genre_image_dataset, val_genre_image_dataset = random_split(genre_image_dataset, lengths,  generator=Generator().manual_seed(42))
    train_genre_image_dataloader, val_genre_image_dataloader = DataLoader(train_genre_image_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_genre_image_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_genre_image_dataloader, val_genre_image_dataloader, genre_from_class_id

def train_to_classify_genres(model, train_gtzan_dataloader, val_gtzan_dataloader, num_epochs=50, learning_rate=0.001, learning_rate_gamma=0.9, model_save_path=None):
    loss = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(opt, gamma=learning_rate_gamma)
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    # initialize file for tqdm progress bar
    progress_file = open(f"progress_{type(model).__name__}.log", "w")

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_gtzan_dataloader, total=len(train_gtzan_dataloader), file=progress_file)
        total_loss = 0
        total_correct = 0
        for batch_no, batch in enumerate(train_gtzan_dataloader):
            mel_spectrogram, genre = batch

            softmax_logits = model(mel_spectrogram)
            cross_entropy_loss = loss(softmax_logits, genre)

            predicted_genre = targmax(softmax_logits, dim=1)
            num_correct = tsum(predicted_genre==genre)

            total_loss += cross_entropy_loss.mean().item()
            total_correct += num_correct.item()

            opt.zero_grad()
            cross_entropy_loss.mean().backward()
            opt.step()

            progress_bar.set_description(f"Model: {type(model).__name__} Epoch: {epoch+1}/{num_epochs}, Batch: {batch_no}, Loss: {cross_entropy_loss.mean().item():.4f}, Accuracy: {num_correct.item()/genre.shape[0]:.4f}")
            progress_bar.update()
        scheduler.step()
        train_loss_history.append(total_loss/len(train_gtzan_dataloader.dataset))
        train_accuracy_history.append(num_correct.item()/len(train_gtzan_dataloader.dataset))
        total_val_loss = 0
        total_val_correct = 0
        for val_batch in val_gtzan_dataloader:
            val_mel_spectrogram, val_genre = val_batch
            val_softmax_logits = model(val_mel_spectrogram)
            val_predicted_genre = targmax(val_softmax_logits, dim=1)
            val_num_correct = tsum(val_predicted_genre==val_genre)
            val_cross_entropy_loss = loss(val_softmax_logits, val_genre)
            total_val_loss += val_cross_entropy_loss.mean().item()
            total_val_correct += val_num_correct.item()
        val_loss_history.append(total_val_loss/len(val_gtzan_dataloader.dataset))
        val_accuracy_history.append(total_val_correct/len(val_gtzan_dataloader.dataset))


    if model_save_path is not None:
        model.save(model_save_path)
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

