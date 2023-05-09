
# Import helper functions
import numpy as np

# Define constants
IMAGE_HEIGHT, IMAGE_WIDTH = 30, 28
NUM_CLASSES = 10
FOLD_SIZE = 120

#################################################
# Functions for evaluating reconstruction error
# and accuracy
#################################################
def accuracy(y_true, y_pred):
    num_correct = 0
    for i in range(len(y_true)):
        true_label  = get_label(y_true[i])
        recon_label = get_label(y_pred[i])

        if true_label == recon_label:
            num_correct += 1

    return num_correct/len(y_true)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


#################################################
# Functions for preprocess MNIST images
#################################################
def process_image(image, labeled: bool=True):
    def convert_to_binary(image):
        return np.array([1 if x >= 255//2 else 0 for x in image])
    
    if labeled:
        label = image[0]
    
    image = image[1:]
    image = convert_to_binary(image)

    image_reshaped = image.reshape(28, 28)
    
    label_pixels = np.zeros((2, 28))
    if labeled:
        label_pixels[:, label] = 1


    labeled_image = np.concatenate([image_reshaped, label_pixels])
    return labeled_image.flatten()

def get_samples(data, label, num_samples:int=10):
    label_idxs = np.argwhere(data.iloc[:, 0].values == label).flatten()[:num_samples]
    samples = data.iloc[label_idxs, :].values

    return samples


def get_label(image):
    image = image.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    label_pixels = image[-2:, :]

    if len(np.argwhere(label_pixels == [[1], [1]])) > 0:
        label = np.argwhere(label_pixels == [[1], [1]])[0][1]
    else:
        label = -1
    return label

def remove_label(image):
    _image = image.copy()
    _image = _image.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    _image[-2:, :] = 0

    return _image.flatten()


#################################################
# Functions that create folds for 
# K-Fold Cross-Validation
#################################################
def create_folds(data, fold_size=FOLD_SIZE):
    folds = []
    # Create 5 folds
    class_size = fold_size//NUM_CLASSES
    for fold in range(5):
        fold_data = []
        for label in range(NUM_CLASSES):
            class_data = get_samples(data, label, num_samples=class_size) # NumPy array for class data for fold
            fold_data.extend(class_data)
        fold_data = np.array(fold_data)
        np.random.shuffle(fold_data)
        folds += [fold_data]
    return folds
