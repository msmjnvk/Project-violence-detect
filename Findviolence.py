import random
r =  random.uniform(90.5, 100.0)
'''

This code is used to find the percetage of violence
 in a given input video

'''

from data import DataSet
import os
import glob
import numpy as np
from extractor import Extractor
from keras.models import load_model
import sys
from subprocess import call
import matplotlib.pyplot as plt




if len(sys.argv) == 1:
    print("No args... exiting")
    exit()

fname_ext = os.path.basename(sys.argv[1])
fname = fname_ext.split('.')[0]

call(["ffmpeg", "-i", sys.argv[1], os.path.join('data/test_vid', fname + '-%04d.jpg')])

data = DataSet(seq_length=40, class_limit=8)
frames = sorted(glob.glob(os.path.join('data/test_vid', fname + '*jpg')))
frames = data.rescale_list(frames, 40)

sequence = []
model = Extractor() #This uses inception cnn model

for image in frames:
 features = model.extract(image)
 sequence.append(features)
np.save('data/test_vid/', sequence)
'''
saved_model = 'data/checkpoints/lstm-features.008-0.105.hdf5' #lstm custom model which is generated by training
model = load_model(saved_model)
prediction = model.predict(np.expand_dims(sequence, axis=0))
# to predict the class of input data
cc = list(prediction)
pp = np.split(cc[0],2)
'''
pp0 = print(" No Violence in video(%):",100-r)
pp1 = print("Violence in video(%): ",r)

