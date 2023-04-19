from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from time import time

# Create an ImageDataGenerator for the training data
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_it = train_datagen.flow_from_directory(
    "dataset_upma_vs_halwa/train/",
    class_mode="binary",
    batch_size=64,
    target_size=(200, 200),
)

# Create an ImageDataGenerator for the test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_it = test_datagen.flow_from_directory(
    "dataset_upma_vs_halwa/test/",
    class_mode="binary",
    batch_size=64,
    target_size=(200, 200),
)

# Define the model architecture
model = Sequential()
model.add(Flatten(input_shape=(200, 200, 3)))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model on the data from the ImageDataGenerator
dir = "tb_callbacks/MLP"
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
    ", Total params: ",
    model.count_params(),
)
