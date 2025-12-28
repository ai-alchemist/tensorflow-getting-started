#from absl import logging
#logging.set_verbosity(logging.FATAL)

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # now import the tensorflow module
#tf.get_logger().setLevel('ERROR')

#import warnings
#warnings.filterwarnings('ignore')

#import numpy

#####################

print("\n\n\n\n----------------------------\n")
print(tf.__version__)  # make sure the version is 2.x


string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

tf.rank(rank2_tensor)
rank2_tensor.shape


tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,3]
                                                                             
# The number of elements in the reshaped tensor MUST match the number in the original





print("\n\nProgram is complete.\n")


