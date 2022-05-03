import tensorflow as tf
import numpy as np

test_image = np.array([1,2,3,4,5,6,7,0])
test_image = test_image[None,:,None,None] # batch, row, col, depth

print(tf.image.extract_patches(test_image,sizes=[1,2,1,1], strides=[1,2,1,1], rates=[1,1,1,1], padding="VALID"))