import pandas as pd
import numpy as np
from scipy.signal import butter,filtfilt

class EEG():
    def __init__(self,signal,label,indiv):
        self.signal = signal #shape (time_length,n_channels)
        self.clean_signal = None
        self.label = label
        self.indiv = indiv
        self.sampling_freq = 250


    def clean(self):
        #2nd order butterworth filter between 1 and 50 hz
        b,a = butter(2,Wn = (1,50),btype='bandpass',fs = self.sampling_freq) #gross filtering
        self.clean_signal = np.zeros(self.signal.shape)
        for i in range(self.signal.shape[0]):
            self.clean_signal[i,:] = filtfilt(b,a,self.signal[i,:])
