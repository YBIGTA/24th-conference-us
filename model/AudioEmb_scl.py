from pydub import AudioSegment
import torch
import numpy as np 
import librosa
import sklearn

def MP32WAV(file):
    # convert MP3 file to WAV form
    track = AudioSegment.from_mp3(file)
    final = track.export(file[:-3] + 'wav', format='wav')
    return final

def M4A2WAV(file):
    # convert M4A file to WAV form
    track = AudioSegment.from_file(file, format='m4a')
    final = track.export(file[:-3] + 'wav', format='wav')
    return final


def normalize(x, axis=0):
  return sklearn.preprocessing.minmax_scale(x, axis=axis)

def feature_ext(audiofile, stt):
    """
    audio embedding vector를 만드는 함수 
    input : audiofile (.mp3, .m4a, .wav)
    output : (40,) np.array(or torch.tensor)
    """
    
    # audio format to WAV
    if audiofile[-3:].upper() == 'MP3':
        audiofile = MP32WAV(audiofile)
    elif audiofile[-3:].upper() == 'M4A':
        audiofile = M4A2WAV(audiofile)
    else:
        pass

    # audio load
    y, sr = librosa.load(audiofile)

    # Zero crossing rate
    zero_crossings = librosa.zero_crossings(y, pad=False)
    zcr = sum(zero_crossings) / len(y)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff = normalize(spectral_rolloff)
    spr_mean = np.mean(spectral_rolloff) 
    spr_var = np.var(spectral_rolloff) * 10
    
    # Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroids = normalize(spectral_centroids)
    spc_mean = np.mean(spectral_centroids)
    spc_var = np.var(spectral_centroids) * 10

    # Chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    chr_mean = np.mean(chromagram, axis=1)
    chr_var = np.var(chromagram, axis=1)
    
    # Harmonic and Percussive Components
    y_harm, y_perc = librosa.effects.hpss(y)
    y_harm = normalize(y_harm)
    y_perc = normalize(y_perc)
    harm_mean = np.mean(y_harm)
    harm_var = np.var(y_harm) * 10
    perc_mean = np.mean(y_perc)
    perc_var = np.var(y_perc) * 10
    
    # Frequency
    Fourier = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) 
    F_var = np.var(Fourier) / 10
    F_dif = (np.max(Fourier) - np.min(Fourier)) / 1000
    
    # Speeching speed
    SS = len(stt.split(' ')) / (len(y)/sr)
    
    final = np.array([zcr, spr_mean, spr_var, spc_mean, spc_var, harm_mean, 
                     harm_var, perc_mean, perc_var, F_var, F_dif, SS])
    final = np.append(final, chr_mean)
    final = np.append(final, chr_var)
    
    # if torch
    # final = torch.Tensor(final)
    
    # final.shape : (36,)
    return final 

