import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from time import time
from keras.callbacks import TensorBoard
from tensorflow import summary


def define_model(blocks):
    assert blocks > 0
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=(200, 200, 3),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    for i in range(blocks - 1):
        model.add(
            Conv2D(
                2 ** (6 + i),
                (3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                padding="same",
            )
        )
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(1, activation="sigmoid"))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


def run_test_harness(blocks, data_aug):
    model = define_model(blocks)
    train_datagen = (
        ImageDataGenerator(
            rescale=1.0 / 255.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        if data_aug
        else ImageDataGenerator(rescale=1.0 / 255.0)
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = train_datagen.flow_from_directory(
        "dataset_upma_vs_halwa/train/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200),
    )
    test_it = test_datagen.flow_from_directory(
        "dataset_upma_vs_halwa/test/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200),
    )
    dir="tb_callbacks/VGG"+str(blocks)+"_data_augmentation_"+str(data_aug)
    tensorboard_callback = TensorBoard(log_dir=dir, histogram_freq=1)
    start = time()
    history = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=test_it,
        validation_steps=len(test_it),
        callbacks=[tensorboard_callback],
        epochs=20,
        verbose=2,
    )
    end = time()
    train_loss, train_acc = model.evaluate(train_it, steps=len(train_it), verbose=0)
    test_loss, test_acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print(
        "Training Loss: %.3f" % train_loss,
        ", Test loss: %.3f" % test_loss,
        ", Training time(in s): %.3f" % (end - start),
        ", Train accuracy: %.3f" % (train_acc * 100),
        ", Test accuracy: %.3f" % (test_acc * 100),
        ", Total params: ",model.count_params()
    )
    summarize_diagnostics(history)

for mod in ((1,False),(2,False),(3,True)):
    run_test_harness(mod[0],mod[1])
