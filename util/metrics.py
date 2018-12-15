
import numpy as np

# jaccard distance
def jaccard(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    # compute jaccard
    intersection = np.sum(np.multiply(x, y))
    union = np.sum(x) + np.sum(y) - intersection
    return intersection / union

# dice coefficient
def dice(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    # compute dice
    intersection = np.sum(np.multiply(x, y))
    return 2*(intersection + eps) / (np.sum(x) + np.sum(y) + eps)

# accuracy related metrics:
#   - accuracy, precision, recall, f-score
def accuracy_metrics(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    tp = np.sum(np.multiply(x, y))
    tn = np.sum(np.multiply(1-x, 1-y))
    fp = np.sum(np.multiply(x, 1-y))
    fn = np.sum(np.multiply(1-x, y))

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_score = 2*(precision*recall)/(precision+recall)

    return accuracy, precision, recall, f_score