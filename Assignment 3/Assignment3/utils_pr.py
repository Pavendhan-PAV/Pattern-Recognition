import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Utils_PR():
    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

    def plot_confusion_matrix(self, y_true, predicted):
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, predicted)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()
