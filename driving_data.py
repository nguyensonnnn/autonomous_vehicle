import cv2
import random
import numpy as np

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * 3.14159265 / 180)

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
#random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.5)]
train_ys = ys[:int(len(xs) * 0.5)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def split_and_shuffle_images(length):
    # Split the image paths into groups of 5
    image_groups = [train_xs[i:i+length] for i in range(0, len(train_xs), length)]
    output_groups=[train_ys[i:i+length] for i in range(0, len(train_ys), length)]
    # Shuffle the image groups
    c = list(zip(image_groups, output_groups))
    random.shuffle(c)
    image_groups, output_groups = zip(*c)
    
    
    return list(image_groups),output_groups

train_xs_new,train_ys_new=split_and_shuffle_images(5)


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        image_group = list(train_xs_new[(train_batch_pointer + i) % (num_train_images // 5)])
        for k in range(len(image_group)):
            image_group[k]=cv2.resize(cv2.imread(image_group [k])[-150:], (200, 66)) / 255.0
        train_xs_new[(train_batch_pointer + i) % (num_train_images // 5)]=tuple(image_group)
        x_out.append(list(train_xs_new[(train_batch_pointer + i) % (num_train_images //5)]))
        y_out.append([train_ys_new[(train_batch_pointer + i) %( num_train_images //5)]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out

#for i in range(2):
data=LoadTrainBatch(100)
print(np.shape(data))