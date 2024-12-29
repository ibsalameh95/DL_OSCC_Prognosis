import argparse
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math 
import os 

parser = argparse.ArgumentParser(description='Plot the loss vs iteration and accuracy vs iteration for given data file')
parser.add_argument('--data_file', default='Results/Results/prognosis_model/validation/validation_results____2024_08_16__18_03_32_56.txt', help='Data file path', dest='data_file')

FLAGS = parser.parse_args()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          current_ax = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        axis_sum = cm.sum(axis=1)[:, np.newaxis]
        axis_sum[axis_sum == 0] = 1
        cm2 = cm.astype('float') / axis_sum
        cm_normalized = (cm2.astype('float') - np.amin(cm2)) / (np.amax(cm2)-np.amin(cm2))
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        cm_normalized = (cm.astype('float') - np.amin(cm)) / (np.amax(cm)-np.amin(cm))

    ax = current_ax
    if normalize:
        im = ax.imshow(cm2, interpolation='nearest', cmap=cmap)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        ax.set_title(title, fontweight="bold")
    # plt.colorbar()
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    # ax.set_xticklabels(classes)
    ax.set_xticklabels(classes, rotation=0, horizontalalignment="right",)
    # ax.set_yticklabels(classes)
    ax.set_yticklabels(classes, rotation=90)
    ax.set_ylim( (len(classes)-0.5, -0.5) )


    fmt = '.2%' if normalize else '.2%'
    thresh = 0.5
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            
            ax.text(j, i, '{}\n({:.2%})'.format(cm[i, j],cm2[i, j]),
                     horizontalalignment="center",verticalalignment="center",
                     fontsize=10,
                     color="white" if cm_normalized[i, j] > thresh else "black")
    else: 
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",verticalalignment="center",
                     fontsize=10, 
                     color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_ylabel('Truth')
    ax.set_xlabel('Predicted')


        # plot confusion matrix:
def plot_cm(results, out_dir, title, class_names):

        label_index_arr = np.asarray(results[:,2],dtype=str)
        pred_arr = np.asarray(results[:,3],dtype=str)

        # label_index_arr = np.asarray(results[:,1],dtype=int)
        # pred_arr_scores = np.asarray(results[:,2],dtype=float)
        # pred_arr = [0 if sc < 0.5 else 1 for sc in pred_arr_scores]

        conf_mat = confusion_matrix(label_index_arr, pred_arr)

        plt.rcParams.update({'font.size': 10})

        fig, ax = plt.subplots(figsize=(6,6))

        plot_confusion_matrix(conf_mat, title=title, classes=class_names, normalize=True, current_ax=ax)
        fig.subplots_adjust(left=0.20, bottom=0.20, right=0.25, top=0.25, wspace=0.2 ,hspace=0.20)
        
        plt.tight_layout()
        plt.savefig(out_dir, dpi=300)
        fig.clear()


data_arr = np.loadtxt(FLAGS.data_file, dtype='str', comments='#', delimiter='\t')

out_file = '{}__confusion_mtx.png'.format(FLAGS.data_file[:-4])
title = None
class_names = ['Good Prog', 'Poor Prog']

plot_cm(data_arr, out_file, None, class_names)
            

# for root, dirs, files in os.walk('Results/prognosis_model/test_metrics/'):
#     for file in files: 
#         if file == 'slide_scores.txt': 
            
#             print(os.path.join(root, file)[:-4])
#             data_arr = np.loadtxt(os.path.join(root, file), dtype='str', comments='#', delimiter='\t')

#             out_file = '{}__confusion_mtx.png'.format(os.path.join(root, file)[:-4])
#             title = None
#             class_names = ['Good Prog', 'Poor Prog']

#             plot_cm(data_arr, out_file, None, class_names)
            




