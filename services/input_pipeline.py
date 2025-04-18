import tensorflow as tf
import preprocess as pre
import numpy as np

dataset=tf.convert_to_tensor(pre.df)

def normalize(dataset):
    
    mean,var=tf.nn.moments(dataset,axes=[0])
    nor_data_ =(dataset-mean)/tf.sqrt(var)
    nor_data=tf.data.Dataset.from_tensor_slices(nor_data_)
    return nor_data

nor_data=normalize(dataset)

assert isinstance(nor_data,tf.data.Dataset)

class input_pipeline:

    def __init__(self,x):
        self.x=x

    def pipe(self):
        x=self.x.shuffle(buffer_size=100)
        x=x.batch(128)
        x=x.prefetch(tf.data.AUTOTUNE)
        return x
    
obj=input_pipeline(nor_data)
x=obj.pipe()

print(x)
