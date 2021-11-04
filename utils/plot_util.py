from matplotlib import pyplot as plt
from os import path, makedirs

import pdb


def draw_plot(logs, save_dir, labels=None):
    """
    logs: losses, accuracy (metrics)
    save_dir: directory to save the plot
    """
    try:
        if not path.exists(save_dir):
            makedirs(save_dir)
    except OSError:
        print ('[Error] fail to create directory: ' +  save_dir)
    
    # plot logs in one figure
    plt.figure()
    for key in labels:  # logs.keys()
        plt.plot(logs[key])
        if labels is not None:
            plt.legend(labels)
    plt.savefig(path.join(save_dir, '_'.join(labels) + '.png'))
    plt.show()