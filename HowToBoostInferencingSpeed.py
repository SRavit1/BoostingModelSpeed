import numpy as np
import tensorflow as tf
import cv2

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

pretrainedMobileNet = tf.keras.applications.MobileNet()

imagePath = "./soccerBall.jpeg"
inputImage = tf.keras.preprocessing.image.load_img("./soccerBall.jpeg", target_size=[224, 224])

inputImage = tf.keras.preprocessing.image.img_to_array(inputImage)
inputImage = tf.keras.applications.mobilenet.preprocess_input(inputImage[tf.newaxis,...])

result = pretrainedMobileNet(inputImage)
decoded = imagenet_labels[np.argsort(result)[0,::-1][:5]+1]

print("This is a " + decoded[0] + "!!")
#cv2.imshow(cv2.imread(imagePath))

import time

startTime = time.time()

numFrames = 100
for i in range(100):
  #print("Running for iteration #" + str(i))
  result = pretrainedMobileNet(inputImage)

elapsedTime = time.time() - startTime
print("This took {} seconds.".format(elapsedTime))
print("Average latency is {} ms.".format(elapsedTime * 1000./numFrames))
print("Average speed is {} frames / second.".format(numFrames/elapsedTime))

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

full_model = tf.function(lambda x:pretrainedMobileNet(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(pretrainedMobileNet.inputs[0].shape, pretrainedMobileNet.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

startTime = time.time()

numFrames = 100
for i in range(100):
  #print("Running for iteration #" + str(i))
  result = frozen_func(tf.constant(inputImage))

elapsedTime = time.time() - startTime
print("This took {} seconds.".format(elapsedTime))
print("Average latency is {} ms.".format(elapsedTime * 1000./numFrames))
print("Average speed is {} frames / second.".format(numFrames/elapsedTime))
