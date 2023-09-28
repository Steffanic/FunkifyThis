import argparse
import librosa
import tensorflow as tf
from torchinfo import summary
from GenreStyleTransfer.data import AudioFolder, get_gtzan_audio_dataloader, load_audio_file
from GenreStyleTransfer.models import BaselineGenreClassifier
from GenreStyleTransfer.plot import plot_histories
from GenreStyleTransfer.train import train_to_classify_genres, train_to_stylize_song
from tensorboard.plugins.hparams import api as hp
from scipy.io import wavfile

def get_model(model, version=None):
    if model == 'baseline':
        if version is not None:
            model = [BaselineGenreClassifier()]
            try:
                model[0].load('model_saves/', version)
            except:
                print(f"Could not load model {model} with version {version}. Using untrained model.")
            model[0].version = version
            return model
        else:
            return BaselineGenreClassifier()
    else:
        raise ValueError(f"Model {model} not recognized. Please choose from baseline.")

NUM_EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
LEARNING_RATE_GAMMA = 0.99

learning_rate_values = [0.001, 0.01, 0.1]
batch_size_values = [8, 16, 32]
time_kernel_size_values = [5, 7, 9]
freq_kernel_size_values = [5, 7, 9]
time_stride_values = [1, 2]
freq_stride_values = [1, 2]
num_conv_layers_values = [3, 4, 5]
num_fc_layers_values = [1, 2, 3]

grid_search_combinations = []
for learning_rate in learning_rate_values:
    for batch_size in batch_size_values:
        for time_kernel_size in time_kernel_size_values:
            for freq_kernel_size in freq_kernel_size_values:
                for time_stride in time_stride_values:
                    for freq_stride in freq_stride_values:
                        for num_conv_layers in num_conv_layers_values:
                            for num_fc_layers in num_fc_layers_values:
                                grid_search_combinations.append({"learning_rate": learning_rate, "batch_size": batch_size, "time_kernel_size": time_kernel_size, "freq_kernel_size": freq_kernel_size, "time_stride": time_stride, "freq_stride": freq_stride, "num_conv_layers": num_conv_layers, "num_fc_layers": num_fc_layers})


genre_from_class_id = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

if __name__=="__main__":

    # Maybe make a hyperparameter dictionary here for searching
    for hyperparam_dict in grid_search_combinations:
        # get the dataloader
        train_gtzan_dataloader, val_gtzan_dataloader, genre_from_class_id = get_gtzan_audio_dataloader(audio_folder_root="D:\GTZAN\Data\genres_original", batch_size=hyperparam_dict['batch_size'], shuffle=True, convert_to_mel=True)

        # create the models for the genre classification task
    
        model = BaselineGenreClassifier(num_conv_layers=hyperparam_dict["num_conv_layers"], num_dense_layers=hyperparam_dict["num_fc_layers"], kernel_size=(hyperparam_dict["freq_kernel_size"], hyperparam_dict["time_kernel_size"]), stride=(hyperparam_dict["freq_stride"], hyperparam_dict["time_stride"]))

        with tf.summary.create_file_writer('logs/').as_default():
            hp.hparams_config(
                hparams=[hp.HParam('learning_rate', hp.RealInterval(0.001, 0.1)),
                    hp.HParam('learning_rate_gamma', hp.RealInterval(0.9, 0.99)),
                    hp.HParam('num_epochs', hp.IntInterval(10, 50)),
                    hp.HParam('batch_size', hp.IntInterval(8,16)),
                    hp.HParam('time_kernel_size', hp.IntInterval(5, 11)),
                    hp.HParam('freq_kernel_size', hp.IntInterval(5, 11)),
                    hp.HParam('time_stride', hp.IntInterval(1, 2)),
                    hp.HParam('freq_stride', hp.IntInterval(1, 2)),
                    hp.HParam('num_conv_layers', hp.IntInterval(3, 5)),
                    hp.HParam('num_fc_layers', hp.IntInterval(1, 3))],
                metrics=[hp.Metric('final_val_accuracy', display_name='Validation Accuracy'), hp.Metric('final_validation_loss', display_name='Validation Loss')],
            )
            hp.hparams(hparams={
                'learning_rate': hyperparam_dict["learning_rate"],
                'learning_rate_gamma': LEARNING_RATE_GAMMA,
                'num_epochs': NUM_EPOCHS,
                'batch_size': hyperparam_dict["batch_size"],
                'time_kernel_size': model.kernel_size[1],
                'freq_kernel_size': model.kernel_size[0],
                'time_stride': model.stride[1],
                'freq_stride': model.stride[0],
                'num_conv_layers': model.num_conv_layers,
                'num_fc_layers': model.num_dense_layers,
            })
        # grab the first batch for a summary
        mel_spectrogram, _, _ = next(iter(val_gtzan_dataloader))

        batch_shape = mel_spectrogram.shape
        
        summary(model, input_size=batch_shape)

        histories = {}
        histories[type(model).__name__] = train_to_classify_genres(model, train_gtzan_dataloader=train_gtzan_dataloader, val_gtzan_dataloader=val_gtzan_dataloader, num_epochs=NUM_EPOCHS, learning_rate=hyperparam_dict['learning_rate'], learning_rate_gamma=LEARNING_RATE_GAMMA, model_save_path='model_saves/')
        with tf.summary.create_file_writer('logs/').as_default():
            tf.summary.scalar('final_val_accuracy', histories[type(model).__name__]['val_accuracy'][-1], step=1)
            tf.summary.scalar('final_validation_loss', histories[type(model).__name__]['val_loss'][-1], step=1)
        # plot the histories
        plot_histories(histories, type(model).__name__, title_string=f"{NUM_EPOCHS=}, {hyperparam_dict['learning_rate']=}, {LEARNING_RATE_GAMMA=}, {hyperparam_dict['batch_size']=}", plot_save_path=f'plots/{hyperparam_dict}.png')

