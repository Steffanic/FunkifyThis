from copy import copy
from torch.nn import CrossEntropyLoss, Parameter
from torch.optim import Adam, lr_scheduler
from torch import argmax as targmax, sum as tsum, mean as tmean, save as tsave, normal as tnormal, stack as tstack, square as tsquare, abs as tabs, clone as tclone
from torch.linalg import matrix_norm as tmatrix_norm
from torch import Tensor
from tqdm import tqdm
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from time import time

from GenreStyleTransfer.data import AudioFolder

def gram(X):
    '''Computes the Gram matrix of X where X is a tensor of shape=(batch_size, num_channels, height, width)'''
    X = X.view(X.shape[1], -1)
    
    return X.matmul(X.transpose(0, 1)) / (X.shape[0]*X.shape[1])

def tv_loss(Y_hat):
    return 0.5 * (tabs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() + tabs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


def train_to_classify_genres(model, train_gtzan_dataloader, val_gtzan_dataloader, num_epochs=50, learning_rate=0.001, learning_rate_gamma=0.9, model_save_path=None):
    loss = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(opt, gamma=learning_rate_gamma)

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    grad_magnitudes = []

    # initialize file for tqdm progress bar
    progress_file = open(f"progress_{type(model).__name__}.log", "w")
    train_writer = tf.summary.create_file_writer(f"runs/train/{time()}/")

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_gtzan_dataloader, total=len(train_gtzan_dataloader), file=progress_file)
        total_loss = 0
        total_correct = 0

        for batch_no, batch in enumerate(train_gtzan_dataloader):
            mel_spectrogram, genre, _ = batch # load the batch

            softmax_logits = model(mel_spectrogram) # forward pass
            cross_entropy_loss = loss(softmax_logits, genre) # compute the loss

            predicted_genre = targmax(softmax_logits, dim=1) # get the predicted genre
            num_correct = tsum(predicted_genre==genre) # compute the number of correct predictions


            opt.zero_grad()
            cross_entropy_loss.mean().backward() # compute the gradients
            opt.step()
            total_loss += cross_entropy_loss.mean().item() # add the loss to the total loss
            total_correct += num_correct.item() # add the number of correct predictions to the total number of correct predictions
            
            grad_magnitudes.append([(name, tmean(param.grad.abs()).item()) for name, param in model.named_parameters()]) # compute the gradient magnitudes and save them
            with train_writer.as_default():
                for name, param in model.named_parameters():
                    tf.summary.histogram(f"grads/grad_magnitude_{name}", tmean(param.grad.abs()), step=epoch*len(train_gtzan_dataloader)+batch_no)

            progress_bar.set_description(f"Model: {type(model).__name__} Epoch: {epoch+1}/{num_epochs}, Batch: {batch_no}, Loss: {cross_entropy_loss.mean().item():.4f}, Accuracy: {num_correct.item()/genre.shape[0]:.4f}")
            progress_bar.update()

        scheduler.step()

        train_loss_history.append(total_loss/batch_no)
        train_accuracy_history.append(num_correct.item()/batch_no)

        with train_writer.as_default():
            tf.summary.scalar("loss/train_loss", total_loss/batch_no, step=epoch)
            tf.summary.scalar("accuracy/train_accuracy", num_correct.item()/batch_no, step=epoch)

        total_val_loss = 0
        total_val_correct = 0
        for batch_no, val_batch in enumerate(val_gtzan_dataloader):
            val_mel_spectrogram, val_genre, _= val_batch
            val_softmax_logits = model(val_mel_spectrogram)
            val_predicted_genre = targmax(val_softmax_logits, dim=1)
            val_num_correct = tsum(val_predicted_genre==val_genre)
            val_cross_entropy_loss = loss(val_softmax_logits, val_genre)
            total_val_loss += val_cross_entropy_loss.mean().item()
            total_val_correct += val_num_correct.item()
        val_loss_history.append(total_val_loss/len(val_gtzan_dataloader.dataset))
        val_accuracy_history.append(total_val_correct/len(val_gtzan_dataloader.dataset))

        with train_writer.as_default():
            tf.summary.scalar("loss/val_loss", total_val_loss/batch_no, step=epoch)
            tf.summary.scalar("accuracy/val_accuracy", total_val_correct/(batch_no*len(val_genre)), step=epoch)

    

    if model_save_path is not None:
        model.save(model_save_path)

    return {"train_loss":train_loss_history, "train_accuracy":train_accuracy_history, "val_loss":val_loss_history, "val_accuracy":val_accuracy_history, "grad_mag":grad_magnitudes}

def train_to_stylize_song(genreClassifier, contentSong, styleSong, style_level, genre_from_class_idx, num_epochs=50, learning_rate=0.001, learning_rate_gamma=0.9, tv_weight = 1, model_save_path=None):
    x_mel_spectrogram = Parameter(Tensor(contentSong).clone())
    loss = CrossEntropyLoss()
    style_opt = Adam([x_mel_spectrogram], lr=learning_rate)

    content_loss_history = []
    style_loss_history = []
    total_loss_history = []

    # initialize file for tqdm progress bar
    progress_file = open(f"progress_{type(genreClassifier).__name__}.log", "w")
    style_writer = tf.summary.create_file_writer(f"runs/style/{time()}/")

    content_mel_spectrogram = Tensor(contentSong).clone()
    style_mel_spectrogram = Tensor(styleSong).clone()
    #x_mel_spectrogram =  content_mel_spectrogram + normal(0, 1, content_mel_spectrogram.shape)
    x_mel_spectrogram = x_mel_spectrogram + tnormal(0, 0.5, x_mel_spectrogram.shape)

    content_logits = genreClassifier(content_mel_spectrogram.unsqueeze(0))
    content_activations = {}
    for key in genreClassifier.content_layer_activations.keys():
        content_activations[key] = genreClassifier.content_layer_activations[key].clone()
    style_logits = genreClassifier(style_mel_spectrogram.unsqueeze(0))
    style_activations = {}
    for key in genreClassifier.style_layer_activations.keys():
        style_activations[key] = genreClassifier.style_layer_activations[key].clone()
        
    for epoch in range(num_epochs):
        progress_bar = tqdm(file=progress_file)
    
        # add some noise
        

        for style_update_i in range(60):

            content_loss = 0
            style_loss = 0

            x_softmax_logits = genreClassifier(x_mel_spectrogram.unsqueeze(0))
            x_content_activations = genreClassifier.content_layer_activations
            x_style_activations = genreClassifier.style_layer_activations

            for key in x_content_activations.keys():
                content_loss += tmean(tsquare(x_content_activations[key]-content_activations[key]))

            for key in x_style_activations.keys():
                style_loss += tmean(tsquare(gram(x_style_activations[key])-gram(style_activations[key])))

            

            total_loss = content_loss + style_level*style_loss 
            with style_writer.as_default():
                tf.summary.scalar("loss/content_loss", content_loss.item(), step=style_update_i)
                tf.summary.scalar("loss/style_loss", style_loss.item(), step=style_update_i)
                tf.summary.scalar("loss/total_loss", total_loss.item(), step=style_update_i)

            content_loss_history.append(content_loss.item())
            style_loss_history.append(style_loss.item())
            total_loss_history.append(total_loss.item())

            style_opt.zero_grad()
            (total_loss).backward(retain_graph=True)
            style_opt.step()
        with style_writer.as_default():
            tf.summary.image("stylized_mel", x_mel_spectrogram.unsqueeze(0).permute(0,2,3,1).detach(), step=epoch)
            tf.summary.audio("stylized_audio", AudioFolder._convert_mel_to_audio(x_mel_spectrogram.unsqueeze(0), sample_rate=22500).transpose(1,2).detach(), sample_rate=22500, step=epoch)

        



        predicted_genre = targmax(x_softmax_logits, dim=1)
        

        progress_bar.set_description(f"Model: {type(genreClassifier).__name__} Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss.mean().item():.4f} Current Genre: {genre_from_class_idx(predicted_genre)}")
        progress_bar.update()


    if model_save_path is not None:
        genreClassifier.save(model_save_path)

    loss_histories = {"content_loss":content_loss_history, "style_loss":style_loss_history, "total_loss": total_loss_history}

    return x_mel_spectrogram, loss_histories
