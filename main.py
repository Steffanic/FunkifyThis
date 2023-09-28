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

NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
LEARNING_RATE_GAMMA = 0.99

genre_from_class_id = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

if __name__=="__main__":
    # handle arguments
    parser = argparse.ArgumentParser(description='Train a model to classify genres. Then use the model to stylize a song.')
    parser.add_argument('--train', action='store_true', help='train the model', default=False)
    parser.add_argument('--stylize', action='store_true', help='stylize a song', default=False)
    parser.add_argument('--model', type=str, help='the model to use', default='baseline')
    parser.add_argument('--content', type=str, help='the content song path')
    parser.add_argument('--style', type=str, help='the style song path')
    parser.add_argument('--style-level', type=float, help='the style level')
    parser.add_argument('--version', type=int, help='the version of the model to use')
    args = parser.parse_args()

    # Maybe make a hyperparameter dictionary here for searching

    if args.train:
        # get the dataloader
        train_gtzan_dataloader, val_gtzan_dataloader, genre_from_class_id = get_gtzan_audio_dataloader(audio_folder_root="D:\GTZAN\Data\genres_original", batch_size=BATCH_SIZE, shuffle=True, convert_to_mel=True)

        # create the models for the genre classification task
        if args.version is not None:
            # loads a previously trained model
            model = get_model(args.model, args.version)
        else:
            model = get_model(args.model)

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
                        hp.HParam('num_fc_layers', hp.IntInterval(1, 3)),
                        hp.HParam('model', hp.Discrete(['baseline']))],
                metrics=[hp.Metric('final_val_accuracy', display_name='Validation Accuracy')],
            )
            hp.hparams(hparams={
                'learning_rate': LEARNING_RATE,
                'learning_rate_gamma': LEARNING_RATE_GAMMA,
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'time_kernel_size': model.kernel_size[1],
                'freq_kernel_size': model.kernel_size[0],
                'time_stride': model.stride[1],
                'freq_stride': model.stride[0],
                'num_conv_layers': model.num_conv_layers,
                'num_fc_layers': model.num_dense_layers,
                'model': args.model,
            })
        # grab the first batch for a summary
        mel_spectrogram, _, _ = next(iter(val_gtzan_dataloader))

        batch_shape = mel_spectrogram.shape
        
        summary(model, input_size=batch_shape)

        histories = {}
        histories[type(model).__name__] = train_to_classify_genres(model, train_gtzan_dataloader=train_gtzan_dataloader, val_gtzan_dataloader=val_gtzan_dataloader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, learning_rate_gamma=LEARNING_RATE_GAMMA, model_save_path='model_saves/')
        with tf.summary.create_file_writer('logs/').as_default():
            tf.summary.scalar('final_val_accuracy', histories[type(model).__name__]['val_accuracy'][-1], step=1)
        # plot the histories
        plot_histories(histories, type(model).__name__, title_string=f"{NUM_EPOCHS=}, {LEARNING_RATE=}, {LEARNING_RATE_GAMMA=}, {BATCH_SIZE=}", plot_save_path='plots/'+args.model+".png")

        # save the models
        model.save('model_saves/')

    if args.stylize:
        # load the models for the genre style transfer task
        model = get_model(args.model, args.version)

        # load the content and style songs
        content_audio, sample_rate = load_audio_file(args.content)
        style_audio, sample_rate = load_audio_file(args.style)

        # convert the audio to mel spectrograms
        content_mel_spectrogram = AudioFolder._convert_sample_to_mel(content_audio, sample_rate)
        style_mel_spectrogram = AudioFolder._convert_sample_to_mel(style_audio, sample_rate)

        # stylize the song
        style_loss_histories = {}
        stylized_mel_spectrogram = {}
        stylized_mel_spectrogram[type(model).__name__], style_loss_histories[type(model).__name__] = train_to_stylize_song(model, content_mel_spectrogram, style_mel_spectrogram, style_level=args.style_level, genre_from_class_idx=genre_from_class_id, num_epochs=10, learning_rate=0.1, learning_rate_gamma=0.99, model_save_path='model_saves/styleTransfers/')
                
        # convert the mel spectrogram to audio
        stylized_ft = librosa.feature.inverse.mel_to_stft(stylized_mel_spectrogram[type(model).__name__].detach().numpy(), sr=sample_rate)
        stylized_audio = librosa.istft(stylized_ft)
        # save the audio
        wavfile.write(f'stylized_{type(model).__name__}.wav', sample_rate, stylized_audio.T)

        # plot the style loss histories
        plot_histories(style_loss_histories, type(model).__name__ , title_string=f"{args.style_level=}, {NUM_EPOCHS=}, {LEARNING_RATE=}, {LEARNING_RATE_GAMMA=}, {BATCH_SIZE=}", plot_save_path='plots/styleTransfers/')


