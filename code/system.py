"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg


def reduce_dimensions(feature_vectors_full, model, mode=0):
    """Use PCA to reduce dimensions

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    mode - a integer (0 or 1) to select option
        If mode = 0, use given train matrix to compute principle components
        and store v into the dictionary.
        If mode = 1, grab v already exists in the dictionary.
    """
    if mode == 0:
        # Compute principle components
        covx = np.cov(feature_vectors_full, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
        model['v'] = v.tolist()
        v = np.fliplr(v)
        pcatrain_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
        return pcatrain_data[:, 1:11]

    else:

        v = np.array(model['v'])
        v = np.fliplr(v)
        pcatest_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), v)
        return pcatest_data[:, 1:11]


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data, 0)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model, 1)
    return fvectors_test_reduced


def classify_page(page, model):
    """Nearest neighbor classifier


    Params:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    # Super compact implementation of nearest neighbour
    x = np.dot(page, train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)
    labels = labels_train[nearest]

    return labels


def correct_errors(page, labels, bboxes, model):
    """Error correction

    Params:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    labels_to_words(labels, bboxes)

    return labels


def labels_to_words(labels, bboxes):
    """Separate labels into words

    Takes labels and box sizes to return list of separated words

    Params:

    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    """
    words_list = []

    counter = 0
    word = labels[0]
    for i in range(1, len(labels)):
        # check if the space between two labels is less than 7
        # if less than 7 then considered a part of a word
        if 7 > bboxes[i, 0] - bboxes[i - 1, 2] > -10:
            # if (labels[i] != '.' and labels[i] != ',' and labels[i] != ':' and labels[i] != '!' and labels[i] !=
            # '?' and labels[i] != ';'):
            word += labels[i]

        # else if larger than 7 then append the current processing word to the list
        else:
            words_list.append(word)
            if counter < 20:
                print(word)
                counter += 1
            # if (labels[i] != '.' and labels[i] != ',' and labels[i] != ':' and labels[i] != '!' and labels[i] != '?'
            # and labels[i] != ';'):
            word = labels[i]
    return words_list
