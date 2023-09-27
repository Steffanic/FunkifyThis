import matplotlib.pyplot as plt

def plot_histories(histories, model_names, plot_save_path=None):
    """
    Plots the loss and accuracy history for every model.
    
    Args:
    histories: A dict of dicts, keys are the model_names and values are the dictionary of histories where the keys are the history names. Acceptable history names are 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'content_loss', 'style_loss', 'total_loss', and 'grad_mag'.
    model_names: a list of model names to iterate through and plot histories for
    """
    plt.figure(figsize=(20, 10))
    for model_name in model_names:
        num_subplots = 3 if 'grad_mag' in histories[model_name].keys() else 2
        for history_name in histories[model_name].keys():
            if history_name in ['train_loss', 'val_loss', 'content_loss', 'style_loss', 'total_loss']:
                plt.subplot(1,num_subplots,1)
                plt.plot(histories[model_name][history_name], label=f'{model_name} {history_name}')
                plt.ylabel('Loss')
            elif history_name in ['train_accuracy', 'val_accuracy']:
                plt.subplot(1,num_subplots,2)
                plt.plot(histories[model_name][history_name], label=f'{model_name} {history_name}')
                plt.ylabel('Accuracy')
            elif history_name=='grad_mag':
                plt.subplot(1,num_subplots,3)
                # grad_magnitudes is an epoch list containing lists corresponding to each parameters average gradient
                # the inner list is a list of tuples where the first element is the parameter name and the second element is the gradient magnitude
                grad_mags = [[grad_mag for _, grad_mag in epoch] for epoch in histories[model_name][history_name]]
                names = [[name for name, _ in epoch] for epoch in histories[model_name][history_name]]
                line_widths = [10/_ for _ in range(1,len(grad_mags)+1, -1)]
                alpha = [1/_ for _ in range(1,len(grad_mags)+1, -1)]
                for i, grad_mag in enumerate(grad_mags):
                    plt.plot(grad_mag, alpha = alpha[i], label=names[i], linewidth=line_widths[i])

                plt.ylabel('Gradient magnitude')
    plt.legend()
    plt.xlabel('Epoch*Batch')
    plt.tight_layout()

    if plot_save_path is not None:
        plt.savefig(plot_save_path)

    plt.show()

def plot_grad_magnitudes(grad_magnitudes, model_names, plot_save_path=None):
    """
    @deprecated:
        Use plot_histories instead.

    Plots the gradient magnitudes for every model.
    
    Args:
    grad_magnitudes: A dict of dicts, keys are the model_names and values are the dictionary of gradient magnitudes where the keys are the parameter names.
    model_names: a list of model names to iterate through and plot histories for
    """
    fig, ax=plt.subplots(1, len(model_names), figsize=(20, 10))
    for i,model_name in enumerate(model_names):
        cur_ax = ax[i] if len(model_names)>1 else ax
        param_names = [name for name, _ in grad_magnitudes[model_name][0]]
        #grad_magnitudes is a list of tuples
        # the outer list is the epochs
        # the inner list is the parameters
        # the tuple is the parameter name and the gradient magnitude
        grad_mags = [[grad_mag for _, grad_mag in epoch] for epoch in grad_magnitudes[model_name]]
        line_widths = [10/_ for _ in range(1,len(grad_mags)+1, -1)]
        
        cur_ax.plot(grad_mags, label=f'{model_name} grad magnitudes')
        cur_ax.set_xticks(range(len(param_names)))
        cur_ax.set_xticklabels(param_names, rotation=90)
        cur_ax.set_xlabel('Epoch*Batch')
        cur_ax.set_ylabel('Gradient magnitude')
        cur_ax.set_title(model_name)
    plt.tight_layout()
    if plot_save_path is not None:
        plt.savefig(plot_save_path)
    plt.show()