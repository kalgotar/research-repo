import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
from scipy import fftpack
from scipy.signal import butter,filtfilt
import pandas as pd
import numpy as np
import json
import ast
import os
import csv
import warnings
warnings.filterwarnings('ignore')

        
def butter_lowpass_filter_old(data, cutoff, fs, order=2):
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y

def dynamic_cutoff(data,fs):
    sec=len(data)/fs
    equals=sec/len(data)
    time = np.linspace(0, sec, len(data), endpoint=True)
    

    sig_noise_fft = scipy.fftpack.fft(data)
    sig_noise_amp = 2 / time.size * np.abs(sig_noise_fft)
    sig_noise_freq = np.abs(scipy.fftpack.fftfreq(time.size, sec/len(data)))    
    signal_amplitude = pd.Series(sig_noise_amp).nlargest(2).round(0).astype(int).tolist()
    

    #Calculate Frequency Magnitude
    magnitudes = abs(sig_noise_fft[np.where(sig_noise_freq >= 0)])
    #Get index of top 2 frequencies
    peak_frequency = np.sort((np.argpartition(magnitudes, -2)[-2:])/sec)
    # print('peak_frequency: ',peak_frequency)
    cutoff = peak_frequency[0]
    return cutoff

def butter_lowpass_filter(data, cutoff, fs, order=2):    
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y


filepath="C:/Public_Data/yes_drone/"
resultpath="C:/"


df_features_all=pd.DataFrame()
for filename in os.listdir(filepath):    
    if filename.endswith('.wav'):
        y_org, fs = librosa.load(filepath+filename)
        y_mono= librosa.to_mono(y_org)
        sr=fs

        ###############################################################################################
        # Calculate the threshold dynamically
        signal_std = np.std(y_mono)
        threshold_factor = 3  # Adjust the factor based on the desired sensitivity
        threshold = threshold_factor * signal_std

        # Check for the presence of noise based on the threshold
        noise_indices = np.where(np.abs(y_mono) < threshold)[0]
        # if len(noise_indices) > 0:
        #     print("Len of Noise:", len(noise_indices), " Noise detected at indices: ", noise_indices)
        # else:
        #     print("No noise detected.")
        # print('-----------------------------------------------')
        ###############################################################################################
        cutoff=dynamic_cutoff(y_mono,fs)
        # print("cutoff: ",cutoff)
        if cutoff >0:
            y = butter_lowpass_filter(y_mono, cutoff, fs)
        else:
            y=y_mono


    
        frame_duration=0.2#0.1###0.025 #frame duration 50ms
        frame_size = round(sr*frame_duration)
        overlap_frame = int(np.floor(0.18*frame_size) )
        no_of_frames =int(np.floor(len(y)/frame_size))

        temp=0
        columns=[f'chroma_feature_{i}' for i in range(0,12)]
        df_features=pd.DataFrame(columns=columns)        
        for i in range(no_of_frames):

            y_chrom=y[temp+1:temp+frame_size]
            chromagram = np.mean(librosa.feature.chroma_stft(y=y_chrom, sr=sr).T,axis=0)            
            new_row=pd.DataFrame([chromagram],columns=columns)
            df_features = pd.concat([df_features,new_row],ignore_index=True)            
            first_underscore=filename.find('_')
            df_features['label'] = 1# drone means 1 and unknow or noise then 0 
            temp=(temp+frame_size-overlap_frame)        
        df_features_all = pd.concat([df_features_all,df_features],ignore_index=True)
df_features_all.to_csv(resultpath+"test_drone_Features.csv")    

########## Step:1 creating features with labels ##########
