
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from vit import ViT


""" Hyperparameters """
hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
# Image area = 200 * 200 = 40000
# Patch area = 25 * 25 = 625
# Number of patches = Image area // Patch area
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])
# Shape of an unflattened patch = 25, 25, 3 (Patch width, Patch height, Number of channels)
# Shape of an flattened patch = 25 * 25 * 3 = 1875
# Shape of the image input divided into patches = 64 (number of patches), Flattened Patch shape (1875)

hp["batch_size"] = 8
hp["lr"] = 1e-4 # (1 * 10^-4 = 0.0001)
hp["num_epochs"] = 20
hp["num_classes"] = 10
hp["class_names"] = ["Barred_Spiral_Galaxies", "Cigar_Shaped_Smooth_Galaxies", "Disturbed_Galaxies", "Edge_On_Galaxies_With_Bulge", "Edge_On_Galaxies_Without_Bulge", "In_Between_Round_Smooth_Galaxies", "Merging_Galaxies", "Round_Smooth_Galaxies", "Unbarred_Loose_Spiral_Galaxies", "Unbarred_Tight_Spiral_Galaxies"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1): # 80 % - Training, 10 % - Validation, 10 % - Testing
    images = shuffle(glob(os.path.join(path, "*", "*.jpg"))) # Shuffling is important so that it doesn't happen that some classes are not included in any of the 3 sets, all sets should have all classes

    # split_size = int(len(images) * split)
    # print(images)
    # print(split_size)
    train_x, valid_x = train_test_split(images, test_size=split, random_state=42) # 90 % - Train, 10 % - Validation
    train_x, test_x = train_test_split(train_x, test_size=split, random_state=42) # From the 90 % in Train, 90 % in Train & 10 % in Test

    return train_x, valid_x, test_x

def process_image_label(path):
    """ Reading images """
    path = path.decode() # Comment it for checking the image shape as there is no decode function for string objects  # this is useful when the function is going through the tensorflow function in the parse function
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image/255.0 # Comment this line to save the patches without normalizing, else the patches would be black
    # print(image.shape) # For printing the shape of images # (200, 200, 3)

    # """ Preprocessing to patches """
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"]) # Patch shape is basically patch width, patch height and patch channels
    patches = patchify(image, patch_shape, hp["patch_size"]) # image to be converted to patches, shape for the patches being generated and the step size, basically if the patch size is 25, after 25 pixels along the width or height of the image, next patch should start so 25 (hp["patch_size"]) is our step size here
    # print(patches.shape) # For printing the shape of patches # (8, 8, 1, 25, 25, 3) : 8*8*1 = 64 patches of shape (25,25,3)

    # For saving the patches images
    # patches = np.reshape(patches, (64, 25, 25, 3))
    # for i in range(64):
    #      cv2.imwrite(f"files/{i}.png", patches[i])

    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)

    # """ Label """
    # print(path)
    class_name = path.split("\\") # To split the path to get the class name
    # print(class_name)
    class_name = path.split("\\")[-2] # We'll grab the name of the class folder from the path which is always the 2nd last in the path of an image
    # C:\\Users\\OMOLP094\\Desktop\\Galaxy-Type-Prediction-With-Vision-Transformer\\galaxy_type_dataset\\Barred_Spiral_Galaxies\\image_7941_5.jpg
    # [C:,Users,OMOLP094,Desktop,Galaxy-Type-Prediction-With-Vision-Transformer,galaxy_type_dataset,Barred_Spiral_Galaxies,image_7941_5.jpg]

    # print(class_name) # To print the class name
    class_idx = hp["class_names"].index(class_name) # To get the index (Numerical label) for the class name
    # print(class_idx) # To print the class index
    class_idx = np.array(class_idx, dtype=np.int32) # Converting class index to int32 datatype array

    return patches, class_idx

def parse(path): # The parse function is designed to process image data and their corresponding labels, transforming them into a format suitable for machine learning models in TensorFlow. 
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp["num_classes"])

    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["num_classes"])

# tf.numpy_function: This TensorFlow function wraps a Python function (in this case, process_image_label) so that it can be used in TensorFlow's computational graph. It allows the integration of arbitrary Python code within a TensorFlow pipeline.
# process_image_label: This is the custom function that processes the input image and extracts both the image data (patches) and the label. 
# [path]: This argument passes the input path to the process_image_label function.
# [tf.float32, tf.int32]: These are the expected output data types of the process_image_label function, with patches being a float32 tensor and labels being an int32 tensor.
# tf.one_hot: This TensorFlow function converts the integer labels into one-hot encoded vectors. This is commonly used in classification problems where the label needs to be represented as a vector of zeros with a single one at the index of the correct class.
# hp["num_classes"]: This indicates the number of classes in the classification problem. The resulting one-hot vector will have this length.

    return patches, labels

def tf_dataset(images, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8) # the prefetch function would grab 8 batches while a particular batch is being processed
    return ds


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Paths """
    dataset_path = "C:\\Users\\OMOLP094\\Desktop\\Galaxy-Type-Prediction-With-Vision-Transformer\\galaxy_type_dataset"
    model_path = os.path.join("files", "model","ViT_model.keras") # In the newer versions of Keras, the model saving format has changed, and it now expects the file path for model checkpoints to have a .keras extension instead of the older .h5 extension.
    # print(model_path)
    csv_path = os.path.join("files", "history", "log.csv")
    # print(csv_path)
    # load_data(dataset_path) # For checking the splits
    # Ensure that the directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    # print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}") # For checking the splits

    # process_image_label(train_x[0]) # For checking the function for an image in the dataset

    train_ds = tf_dataset(train_x, batch=hp["batch_size"])
    valid_ds = tf_dataset(valid_x, batch=hp["batch_size"])

    # for x, y in train_ds:
    #     print(x.shape, y.shape) # (8, 64, 1875) - Batch size - 8, num of patches in an image = 64, patch shape = 25*25*3 = 1875, (8, 10) - Batch size - 8, num of classes = 10 (for every image - the label would be a one hot encoded vector containing 10 values with each value being 0 except 1 value for the class indexes to which the image doesn't belongs, the 1 value which is not zero will be 1 indicating the numerical label (class index) to which that particular image belongs

    """ Model """
    model = ViT(hp)
    model.compile(
        loss="categorical_crossentropy", # for Multiclass classification
        optimizer=tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
        metrics=["acc"] # Accuracy is the main metric
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True), # Saves the model weights when the validation loss starts reducing
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1), # decreases learning rate when validation loss starts decreasing
        CSVLogger(csv_path), # Saves the training logs
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False), # If the validation loss doesn't decreases for continuous 50 epoch, this will stop the training
    ]

    model.fit(
        train_ds,
        epochs=hp["num_epochs"],
        validation_data=valid_ds,
        callbacks=callbacks
    )

    ## ...
