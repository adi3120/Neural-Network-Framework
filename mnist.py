#import here
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from ANN import *

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images=train_images[:500]
train_labels=train_labels[:500]


train_images = train_images / 255.0
test_images = test_images / 255.0

def resize_images(images):
    resized_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        resized_img = pil_img.resize((20, 20), Image.BILINEAR)
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)

train_images_resized = train_images
test_images_resized = test_images


train_images = train_images_resized.reshape(train_images.shape[0], -1)
test_images = test_images_resized.reshape(test_images.shape[0], -1)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


eta=0.0001

i=InputLayer(train_images.shape[1],"none")

h1=HiddenLayer(10,"relu")
h1.attach_after(i)
h1.set_weights("uniform_random")
h1.set_biases("uniform_random")

h2=HiddenLayer(10,"relu")
h2.attach_after(h1)
h2.set_weights("uniform_random")
h2.set_biases("uniform_random")

o=OutputLayer(train_labels.shape[1],"softmax","crossentropy")
o.attach_after(h2)
o.set_weights("uniform_random")
o.set_biases("uniform_random")

ANN=[i,h1,h2,o]


ANN,loss=gradient_descent_epoch(ANN,train_images,train_labels,eta,1000)
plt.plot(loss)
plt.show()

def test_network(ANN, x_test, y_test, num_samples=20):
    correct_predictions = 0
    for i in range(num_samples):
        input_data = x_test[i]
        actual_label = np.argmax(y_test[i])

        ANN[0].put_values(input_data)
        for layer in ANN:
            layer.forward()

        output = ANN[-1].output()
        predicted_label = np.argmax(output)

        if predicted_label == actual_label:
            correct_predictions += 1

        print(f"Sample {i + 1}: Predicted Label - {predicted_label}, Actual Label - {actual_label}")

    accuracy = correct_predictions / num_samples * 100.0
    print(f"\nAccuracy on {num_samples} test samples: {accuracy:.2f}%")

test_network(ANN, test_images, test_labels, num_samples=100)