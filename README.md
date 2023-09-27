# FunkifyThis

### Inspired by visual neural style transfer, I thought I would take a stab at aural style transfer! 

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
