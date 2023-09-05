from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

from torch import argmax as targmax, sum as tsum, save as tsave

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
        plt.plot(loss_history[model_name], label=f'{model_name} loss')
        plt.xlabel('Epoch*Batch')
        plt.ylabel('Loss')
        plt.title('Loss history')
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_history[model_name], label=f'{model_name} accuracy')
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
    genre_image_dataloader = DataLoader(genre_image_dataset, batch_size=batch_size, shuffle=shuffle)
    return genre_image_dataloader, genre_from_class_id

def train_to_classify_genres(model, gtzan_dataloader, num_epochs=50, learning_rate=0.001, learning_rate_gamma=0.9, model_save_path=None):
    loss = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(opt, gamma=learning_rate_gamma)
    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        progress_bar = tqdm(gtzan_dataloader)
        for batch_no, batch in enumerate(gtzan_dataloader):
            mel_spectrogram, genre = batch

            softmax_logits = model(mel_spectrogram)
            cross_entropy_loss = loss(softmax_logits, genre)

            predicted_genre = targmax(softmax_logits, dim=1)
            num_correct = tsum(predicted_genre==genre)

            loss_history.append(cross_entropy_loss.mean().item())
            accuracy_history.append(num_correct.item()/genre.shape[0])

            opt.zero_grad()
            cross_entropy_loss.mean().backward()
            opt.step()

            progress_bar.set_description(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_no}, Loss: {cross_entropy_loss.mean().item():.4f}, Accuracy: {num_correct.item()/genre.shape[0]:.4f}")
        scheduler.step()
    if model_save_path is not None:
        model_save_path = model_save_path[:-4] + f'_{int(time())}.pth'
        tsave(model.state_dict(), model_save_path)
    return loss_history, accuracy_history

