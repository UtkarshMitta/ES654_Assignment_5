# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
from os import listdir, makedirs
from numpy import asarray, save, load
from random import seed, random
from keras.utils import load_img, img_to_array
from shutil import copyfile

# define location of dataset
folder = "images/upma/"
# plot first few images
for i in range(1, 10, 1):
    # define subplot
    pyplot.subplot(330 + i)
    # define filename
    filename = folder + str(i) + ".jpg"
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()

# define location of dataset
photos, labels = list(), list()
for food_items in ["upma/", "halwa/"]:
    folder = "images/" + food_items
    # enumerate files in the directory
    # determine class
    output = 1.0
    if food_items == "halwa/":
        output = 0.0
        for file in listdir(folder):
            # load image
            photo = load_img(folder + file, target_size=(200, 200))
            # convert to numpy array
            photo = img_to_array(photo)
            # store
            photos.append(photo)
            labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save("upma_vs_halwa_photos.npy", photos)
save("upma_vs_halwa_labels.npy", labels)

photos = load("upma_vs_halwa_photos.npy")
labels = load("upma_vs_halwa_labels.npy")
print(photos.shape, labels.shape)

dataset_home = "dataset_upma_vs_halwa/"
subdirs = ["train/", "test/"]
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ["upma/", "halwa/"]
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.2
# copy training dataset images into subdirectories
for food in ["upma/", "halwa/"]:
    src_directory = "images/" + food
    for file in listdir(src_directory):
        src = src_directory + file
        dst_dir = "train/"
        if random() < val_ratio:
            dst_dir = "test/"
        if food == "upma/":
            dst = dataset_home + dst_dir + "upma/" + file
            copyfile(src, dst)
        elif food == "halwa/":
            dst = dataset_home + dst_dir + "halwa/" + file
            copyfile(src, dst)
