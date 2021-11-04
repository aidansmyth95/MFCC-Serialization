
import matplotlib as mpl
import matplotlib as matplotlib
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import librosa
import librosa.display


def plot_spec_serialization(spec_data, hop_length, start_ms, end_ms, fs=16000):

    fig = plt.figure()
    ax = plt.axes()

    spec_fig = librosa.display.specshow(spec_data, hop_length=hop_length, x_axis='time', sr=fs, ax=ax)
    plt.colorbar(spec_fig, ax=ax)

    ax.axvspan(start_ms/1000, end_ms/1000, color='green', alpha=0.5)
    plt.title('Serialization segment highlighted green')
    plt.tight_layout()
    plt.show()
    plt.cla()
    plt.clf
    plt.close('all')


def plot_spectrogram(spec_data, hop_length, fs=16000):
    fig = plt.figure()

    librosa.display.specshow(spec_data, hop_length=hop_length, x_axis='time', sr=fs)
    plt.colorbar()
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    plt.cla()
    plt.clf
    plt.close('all')

def plot_wav(wav, fs=16000):

    plt.figure()
    librosa.display.waveshow(wav, sr=fs)
    plt.title('Waveform')
    plt.tight_layout()
    plt.show()
    plt.cla()
    plt.clf
    plt.close('all')