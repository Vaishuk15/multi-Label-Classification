import os
import random
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import torch


IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


filenames = os.listdir("/training_v/")
categories = []
category_names = []
for filename in filenames:
    category = filename.split('_')[0]
    categories.append(category)
    if(category not in category_names): category_names.append(category)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

print(category_names)


df.head()


df.tail()


df['category'].value_counts().plot.bar()


sample = random.choice(filenames)
image = load_img("/training_v/"+sample)
plt.imshow(image)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

# First hidden Layer 
# Convolution layers are used to detect patterns such as edges, textures, and shapes in images
#32 will be number of filters, 3*3 will be size of matrix that used detect edges such as vertical etc
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
# Pooling reduces the size of the dimensions of the image with max value
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Second hidden Layer

#64 number of filters, 3*3  will be size of matrix that used detect edges such as face etc
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Third hidden Layer
#128 number of filters, 3*3  will be size of matrix that used detect edges such as face etc
#filters doubles on every layer to have more complex pattern found on pass of each layer
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Dense layer with 512 units has a high capacity to learn complex patterns and representations.
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
# output layer
model.add(Dense(len(category_names), activation='softmax'))

# optimizer adjusts the learning rate during training
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


earlystop = EarlyStopping(patience=10)


# reduce learning rate when accuracy is not improving
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)


callbacks = [earlystop, learning_rate_reduction]


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


train_df['category'].value_counts().plot.bar()
# validate_df['category'].value_counts().plot.bar()


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=32

print(total_train)
print(total_validate)


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/training_v/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


print(train_df['category'])
print(validate_df['category'].dtype)


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/training_v/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/training_v/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# epochs=3 if FAST_RUN else 5
epochs= 2
history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


model.save_weights("fruits.weights.h5")


history.history.keys()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


test_filenames = os.listdir("C:/Users/VaishnaviKanagaraj/Desktop/testing/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# test_df.head()
nb_samples
print(nb_samples)


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "C:/Users/VaishnaviKanagaraj/Desktop/testing/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


predict = model.predict(test_generator)


test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'].value_counts()


test_df['category'].value_counts().plot.bar() 


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("C:/Users/VaishnaviKanagaraj/Desktop/testing/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('fruit_submission.csv', index=False)


from keras.preprocessing import image
def load_and_preprocess_image(img_path, target_size):
    # Load the image
    img = image.load_img(img_path, target_size=target_size)
    # Convert the image to array
    img_array = image.img_to_array(img)
    # Rescale the image
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape (1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Path to the single image you want to predict
img_path = 'C:/Users/VaishnaviKanagaraj/Desktop/testing/test1_a.jpg'

# Load and preprocess the image
img_array = load_and_preprocess_image(img_path, target_size=IMAGE_SIZE)

# Make a prediction
prediction = model.predict(img_array)

# Convert prediction to class label
predicted_class = np.argmax(prediction, axis=1)

# # Assuming you have a dictionary to map class indices to class names
# class_indices = {0: 'class_name_0', 1: 'class_name_1', ..., 14: 'class_name_14'}  # Modify this as per your classes
# predicted_label = class_indices[predicted_class[0]]

# print(f'Predicted label: {predicted_label}')



predict_labels = (prediction > 0.5).astype(int)
print(predict_labels)


print(predicted_class[0])
predicted_probability = prediction[0][predicted_class[0]]
print(predicted_probability)


for index in range(len(category_names)):
    print(index, category_names[index], prediction[0][index])


category_names[predicted_class[0]]