import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


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


def run_test_harness(blocks):
    model = define_model(blocks)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory(
        "dataset_upma_vs_halwa/train/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200),
    )
    test_it = datagen.flow_from_directory(
        "dataset_upma_vs_halwa/test/",
        class_mode="binary",
        batch_size=64,
        target_size=(200, 200),
    )
    history = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=test_it,
        validation_steps=len(test_it),
        epochs=20,
        verbose=0,
    )
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print("> %.3f" % (acc * 100.0))
    summarize_diagnostics(history)


blocks = int(input("Enter the number of blocks: "))
run_test_harness(blocks)
