import matplotlib.pyplot as plt
import os
import librosa as lr
import util
import numpy as np
import pandas as pd
import Graph0
from time import time
from pprint import pprint

def _level2prob(x):
    return float(x - 1) / 4

class Data_provider():
    #This Class loads the meta-data and audio files from the given db path
    #and provides the data in batches to the main prorgamm

    #Constants
    SAMPLERATE = 22050
    FFT_WINDOW_WIDTH = 2048             #Window with for single fft spec
    FFT_STEP = 512                      #Window step size
    SAMPLE_SIZE = 32                    #Sample
    SAMPLE_STEP = 8

    #Variables
    _db_path = ""
    _data_frame = None                  #meta data
    _y_audio = None                     #Audio data of current file
    _melspec = None                     #melspec
    _pointer_file = None                #index of current audio file in db
    _smpls_size = 0                     #number of samples in file (per channel)
    _file_grinding = 0.0                #grinding level of current audio file
    _pointer_sample = None              #pointer on melspec
    _is_stereo = None                   #whether the current file is stereo

    def read_meta_data(self):
        path = self._db_path + "/meta.csv"
        print("Loading meta: " + path)
        self._data_frame = pd.read_csv(path)
        pprint(self._data_frame)



    def get_train_batch(self, BATCHSIZE = 100):
        images = []
        labels = []
        while len(images) != BATCHSIZE:
            sample = self.next_sample()
            if isinstance(sample, int) and sample == -1:  #return False, if at end of db
                if not self.next_melspec():
                    return False
            else:
                img, noise = sample
                images.append(img)
                labels.append(noise)
        #Convert to numpy arrays:
        images, labels = np.array(images, np.float32), np.array(labels, np.float32)
        return images, labels


    def get_eval_batch(self, BATCHSIZE = 100):
        pass

    def get_test_batch(self):
        pass

    def _next_train_idx(self):
        #returns the index of the first trainset file after self._pointer_idx
        #if pointer is already at last pos, returns -1
        idx = self._pointer_file
        while True:
            idx += 1
            if idx >= len(self._data_frame):
                return -1
            if self._data_frame.iloc[idx]["Type"] == "train":
                return idx

    def __init__(self, db_path):
        self._db_path = db_path
        self.read_meta_data()
        self._pointer_file = -1
        self._pointer_sample = 0
        self._is_stereo = False
        self.next_melspec()

    def next_sample(self):
        #Select Subsample
        pos = self._pointer_sample

        #Stopping condition
        if pos == self._smpls_size:
            return -1

        sample = self._melspec[::-1, pos*self.SAMPLE_STEP:pos*self.SAMPLE_STEP + self.SAMPLE_SIZE]

        #If sample too short or file finished
        if len(sample) != 128 or len(sample[0]) != self.SAMPLE_SIZE:
            return -1

        #Normalize
        offset = np.min(sample)
        scale = np.max(sample) - offset
        sample = (sample - offset) / scale

        #Save and return
        self._pointer_sample += 1
        return sample, self._file_grinding

    def _next_file(self):
        idx = self._pointer_file
        next = self._next_train_idx()
        if next <= idx:                #Exit Condition
            return False

        data_row = self._data_frame.iloc[next]
        path = self._db_path + "/" + data_row[1] + "/" + data_row[0]
        y, sr = lr.load(path)

        if(len(y)) == 2:
            self._is_stereo = True
            length = len(y[0])
        else:
            self._is_stereo = False
            length = len(y)

        self._y_audio = y
        self._smpls_size = (length // 512) // 8 - 3
        self._file_grinding = _level2prob(data_row[3])
        self._pointer_file = next
        print("Loading file " + data_row[0] + " with " + str(self._smpls_size) + " samples")
        return True

    def next_melspec(self):
        if self._is_stereo:#Take the second channel for a melspec
            audio = self._y_audio[1]
            self._is_stereo = False
        else:               #Load new file
            success = self._next_file()
            if not success:
                return False
            if self._is_stereo:#stereo
                audio = self._y_audio[0]
            else:               #mono
                audio = self._y_audio

        self._pointer_sample = 0

        #create melspec
        mel = lr.feature.melspectrogram(audio, self.SAMPLERATE, n_fft=self.FFT_WINDOW_WIDTH, hop_length=self.FFT_STEP)
        mel = np.log(mel)
        self._melspec = mel
        return True #Success


if __name__ == "__main__":

    #Initialization
    dp = Data_provider("/home/mknopp/Dropbox/00_BA/Sound_DB")

    #Plot of FFT Samples
    nrows = 5
    ncols = 10
    fig, ax = plt.subplots(nrows=nrows, ncols=nrows, sharex="all", sharey="all")
    result = dp.get_train_batch(BATCHSIZE=nrows*ncols)
    print(result)

    if isinstance(result, int) and result == -1:
        print("Reached end of db")
        exit()
    else:
        imgs = result[0]
        labs = result[1]
    for idx, (img, lab) in enumerate(zip(imgs, labs)):
        plt.subplot(nrows, ncols, idx+1)
        plt.title(str(lab))
        plt.imshow(img, cmap=plt.get_cmap("gray"))
    plt.show()
