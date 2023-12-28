# FunkifyThis

### Inspired by visual neural style transfer, I thought I would take a stab at aural style transfer! Here's the nightmare fuel it makes so far attempting to style a hip hop track as classical:


https://github.com/Steffanic/FunkifyThis/assets/38746732/5b5f85a7-56fd-40a0-9181-74cbe209f821


### Clearly the style transfer did not work out well. But the genre classifier itself does decently. I tested both a baseline standard CNN and one that only convoluted the frequency dimension. Here are the training plots:

#### Baseline CNN

![baseline_genre_classifier_1693949318](https://github.com/Steffanic/FunkifyThis/assets/38746732/7abc86e1-4e2f-484d-8ee5-5a811907d220)

#### Frequency Convolution Only

![freq_conv_only_genre_classifier_1693951027](https://github.com/Steffanic/FunkifyThis/assets/38746732/a6818771-6e24-40ea-9fe5-b4908b76eda1)


## The Plan:
---------------
 - Train a model to classify a song's genre based on its Mel-scaled spectrogram to a satisfactory level
 - Use that model to style a song according to the same ideas from the neural style transfer paper [arxiv](https://arxiv.org/abs/1508.06576)

## First Attempts:
---------------
 - Baseline model: Simple CNN without Pooling
   - Will try to make a table of parameters and performance
  
## To-dos:
---------------
 - Determine evaluation metric for stylizing the song
    - Potentially just using the classifier itself would work
 - Build out experimental design
   - Hyperparameter search
   - Construct visualizations
 - Try more models
