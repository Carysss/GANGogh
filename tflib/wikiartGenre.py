""" Creates batches of images to feed into the training network conditioned by genre, uses upsampling when creating batches to account for uneven distributuions """


import numpy as np
import scipy.misc
import time
import random
import os
#Set the dimension of images you want to be passed in to the network
DIM = 64

#Set your own path to images
path = os.path.normpath('/home/cop98/dev.git/offsite/GANGogh/smallimage/')

#This dictionary should be updated to hold the absolute number of images associated with each genre used during training

styles = {
    'abstract': 1000,
    'animal-painting': 1798,
    'cityscape': 2000,
    'figurative': 1000,
    'flower-painting': 1000,
    'genre-painting': 1000,
    'landscape': 1000,
    'marina': 1000,
    'mythological-painting': 1000,
    'nude-painting-nu': 1000,
    'portrait': 1000,
    'religious-painting': 1000,
    'still-life': 1000,
    'symbolic-painting': 1000
}

# styles = {
#         'abstract': 611,
#         'cityscape': 641,
#         'figurative': 610,
#         'landscape': 651,
#         'portrait': 600,
#         }

# Full dataset
# styles = {
#     'abstract': 14999,
#     'animal-painting': 1798,
#     'cityscape': 6598,
#     'figurative': 4500,
#     'flower-painting': 1800,
#     'genre-painting': 14997,
#     'landscape': 15000,
#     'marina': 1800,
#     'mythological-painting': 2099,
#     'nude-painting-nu': 3000,
#     'portrait': 14999,
#     'religious-painting': 8400,
#     'still-life': 2996,
#     'symbolic-painting': 2999
# }


styleNum = {
    'abstract': 0,
    'animal-painting': 1,
    'cityscape': 2,
    'figurative': 3,
    'flower-painting': 4,
    'genre-painting': 5,
    'landscape': 6,
    'marina': 7,
    'mythological-painting': 8,
    'nude-painting-nu': 9,
    'portrait': 10,
    'religious-painting': 11,
    'still-life': 12,
    'symbolic-painting': 13
    }

curPos = {
    'abstract': 0,
    'animal-painting': 0,
    'cityscape': 0,
    'figurative': 0,
    'flower-painting': 0,
    'genre-painting': 0,
    'landscape': 0,
    'marina': 0,
    'mythological-painting': 0,
    'nude-painting-nu': 0,
    'portrait': 0,
    'religious-painting': 0,
    'still-life': 0,
    'symbolic-painting': 0
    }


testNums = {}
trainNums = {}

#Generate test set of images made up of 1/20 of the images (per genre)
for k,v in styles.items():
    # put a twentieth of paintings in here
    nums = range(v)
    random.shuffle(list(nums))
    testNums[k] = nums[0:v//20]
    trainNums[k] = nums[v//20:]


def get_style(style_number):
    name = 'unknown'
    for style_str, style_num in styleNum.items():
        if style_num == style_number:
            name = style_str
            break

    return name

def inf_gen(gen):
    while True:
        for (images,labels) in gen():
            yield images,labels
            
    

def make_generator(files, batch_size, n_classes):
    if batch_size % n_classes != 0:
        raise ValueError("batch size must be divisible by num classes")

    class_batch = batch_size // n_classes

    generators = []
    
    def get_epoch():

        while True:

            images = np.zeros((batch_size, 3, DIM, DIM), dtype='int32')
            labels = np.zeros((batch_size, n_classes))
            n=0
            for style in styles:
                styleLabel = styleNum[style]
                curr = curPos[style]
                for i in range(class_batch):
                    if curr == styles[style]:
                        curr = 0
                        random.shuffle(list(files[style]))
                    t0=time.time()
                    image = scipy.misc.imread("{}/{}/{}.png".format(path, style, str(curr)),mode='RGB')
                    #image = scipy.misc.imresize(image,(DIM,DIM))
                    images[n % batch_size] = image.transpose(2,0,1)
                    labels[n % batch_size, int(styleLabel)] = 1
                    n+=1
                    curr += 1
                curPos[style]=curr

            #randomize things but keep relationship between a conditioning vector and its associated image
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            yield (images, labels)
                        

        
    return get_epoch

def load(batch_size, class_size):
    return (
        make_generator(trainNums, batch_size, class_size),
        make_generator(testNums, batch_size, class_size),
    )

#Testing code to validate that the logic in generating batches is working properly and quickly
if __name__ == '__main__':
    train_gen, valid_gen = load(100)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        a,b = batch
        print(str(time.time() - t0))
        if i == 1000:
            break
        t0 = time.time()
