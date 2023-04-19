# vgg16 model used for transfer learning on the upma and halwa dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from time import time
from keras.callbacks import TensorBoard

# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    output = Dense(1, activation="sigmoid")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
    # save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory(
        "dataset_upma_vs_halwa/train/",
        class_mode="binary",
        batch_size=64,
        target_size=(224, 224),
    )
    test_it = datagen.flow_from_directory(
        "dataset_upma_vs_halwa/test/",
        class_mode="binary",
        batch_size=64,
        target_size=(224, 224),
    )
    # fit model
    dir="tb_callbacks/VGG16_transfer_learning"
    tensorboard_callback = TensorBoard(log_dir=dir, histogram_freq=1)
    start = time()
    history = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=test_it,
        validation_steps=len(test_it),
        epochs=20,
        callbacks=[tensorboard_callback],
        verbose=1,
    )
    end = time()
    # evaluate model
    train_loss, train_acc = model.evaluate(train_it, steps=len(train_it), verbose=0)
    test_loss, test_acc = model.evaluate(test_it, steps=len(test_it), verbose=0)

    print(
        "Training Loss: %.3f" % train_loss,
        ", Test loss: %.3f" % test_loss,
        ", Training time(in s): %.3f" % (end - start),
        ", Train accuracy: %.3f" % (train_acc * 100),
        ", Test accuracy: %.3f" % (test_acc * 100),
        ", Total params: ",model.count_params(),
    )
    # learning curves
    summarize_diagnostics(history)


# entry point, run the test harness
run_test_harness()
