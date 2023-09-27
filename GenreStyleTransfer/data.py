import librosa
import numpy as np
from torch import Generator, Tensor
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import ToTensor, Compose, Grayscale
from torch.utils.data import DataLoader, random_split
import soundfile as sf



class AudioFolder(DatasetFolder):
    """A dataset class for the GTZAN dataset that returns the audio files"""
    def __init__(self, root, transform=None, target_transform=None, loader=None, convert_to_mel=False):
        '''
        Args:
            root (string): Root directory path.
            transform (callable, optional): A function/transform that takes in an array of audio samples
                and returns a transformed version. E.g, ``transforms.ToTensor``
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            loader (callable, optional): A function to load an audio file given its path.
            convert_to_mel (bool, optional): If True, converts the audio to a mel spectrogram
        '''
        super().__init__(root, loader, ['.wav'], transform, target_transform)
        self.loader = loader
        self.extensions = ['.wav']
        self.convert_to_mel = convert_to_mel
        

    def _get_target(self, path):
        """Returns the target for the given path"""
        return path.split('\\')[-2]
    
    @classmethod
    def _convert_sample_to_mel(cls, sample, sample_rate):
        audio_ft = librosa.stft(sample, n_fft=2048, hop_length=512)

        # dumb easy solution, set desired length to be the length of the first audio file
        if not hasattr(cls, 'desired_sample_length'):
            cls.desired_sample_length = audio_ft.shape[1]
        # subtract the desired length from the current length
        # if the difference is positive, we need to truncate
        # if the difference is negative, we need to pad
        difference = audio_ft.shape[1] - cls.desired_sample_length
        if difference>0:
            audio_ft = audio_ft[:, :cls.desired_sample_length]
        elif difference<0:
            audio_ft = np.pad(audio_ft, ((0,0), (0, -difference)))

        power_spectrum = np.abs(audio_ft)**2
        spectrogram = librosa.feature.melspectrogram(S=power_spectrum, sr=sample_rate)
        spectrogram = Tensor(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)
        return spectrogram
    
    @classmethod
    def _convert_mel_to_audio(cls, mel, sample_rate):
        mel = mel.squeeze(0)
        audio_ft = librosa.feature.inverse.mel_to_stft(mel.detach().numpy(), sr = sample_rate)
        audio = librosa.istft(audio_ft)
        audio = Tensor(audio)
        audio = audio.unsqueeze(0)
        return audio

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (audio, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        try:
            audio, sample_rate  = self.loader(path)
        except:
            print(path)
            print("Oh well 😜")
            #return the last sample
            return self.__getitem__(index-1)
        
        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.convert_to_mel:
            audio = self._convert_sample_to_mel(audio, sample_rate)
        return audio, target, sample_rate
    
def get_gtzan_image_dataloader(image_folder_root, batch_size=64, shuffle=True):
    """Returns a dataloader for the GTZAN dataset."""   
    transforms = Compose([Grayscale(num_output_channels=1), ToTensor()])

    genre_image_dataset = ImageFolder(image_folder_root, transform=transforms)

    genre_from_class_id = lambda idx:[key for key in genre_image_dataset.class_to_idx.keys() if genre_image_dataset.class_to_idx[key]==idx][0]

    val_len = int(len(genre_image_dataset)*0.2)
    train_len = len(genre_image_dataset) - val_len
    lengths = [train_len, val_len]
    
    train_genre_image_dataset, val_genre_image_dataset = random_split(genre_image_dataset, lengths,  generator=Generator().manual_seed(42))

    train_genre_image_dataloader, val_genre_image_dataloader = DataLoader(train_genre_image_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_genre_image_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_genre_image_dataloader, val_genre_image_dataloader, genre_from_class_id

def get_gtzan_audio_dataloader(audio_folder_root, batch_size=64, shuffle=True, convert_to_mel=False):
    '''Returns a dataloader for the GTZAN dataset.'''

    genre_audio_dataset = AudioFolder(audio_folder_root, loader=sf.read, convert_to_mel=convert_to_mel)

    genre_from_class_id = lambda idx:[key for key in genre_audio_dataset.class_to_idx.keys() if genre_audio_dataset.class_to_idx[key]==idx][0]

    val_len = int(len(genre_audio_dataset)*0.2)
    train_len = len(genre_audio_dataset) - val_len
    lengths = [train_len, val_len]

    train_genre_audio_dataset, val_genre_audio_dataset = random_split(genre_audio_dataset, lengths,  generator=Generator().manual_seed(42))

    train_genre_audio_dataloader, val_genre_audio_dataloader = DataLoader(train_genre_audio_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_genre_audio_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_genre_audio_dataloader, val_genre_audio_dataloader, genre_from_class_id

def load_audio_file(path):
    '''Loads an audio file from a given path.'''
    audio, sample_rate = sf.read(path)
    return audio, sample_rate