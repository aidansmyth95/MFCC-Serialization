import numpy as np
import librosa

from visualization import plot_wav, plot_spec_serialization, plot_spectrogram


def serialize_spectrograms_from_wav_files(input_flist, fs=16000):

    serialized_data = []

    frame_size=ms_to_samples(32, fs=fs)
    hop_length = ms_to_samples(16, fs=fs)

    for fN in input_flist:

        wav_data, sr = librosa.load(fN, sr=fs)

        # plot waveform
        plot_wav(wav_data, fs=fs)

        # create mfcc
        mfcc = wav_to_mfcc(wav_data, frame_size=frame_size, hop_length=hop_length, n_mfcc=13, fs=fs)

        # plot spectrogram
        plot_spectrogram(mfcc, hop_length=hop_length, fs=fs)

        # serialize the spectrogram
        spec_slices, start_ms, end_ms = serialize_spectrogram(mfcc, serialization_window_ms=128, serialization_hop_size_ms=96, frame_hop_size_ms=16)
        serialized_data += spec_slices

        # for each slice
        for slice_idx in range(len(spec_slices)):

            # plot the serialization on top of original spectrogram
            plot_spec_serialization(mfcc, hop_length=hop_length, start_ms=start_ms[slice_idx], end_ms=end_ms[slice_idx], fs=fs)

            # plot the slice by itself
            plot_spectrogram(spec_slices[slice_idx], hop_length=hop_length, fs=fs)


    return serialized_data


# define the Librosa MFCC function here
def wav_to_mfcc(input_wav, frame_size, hop_length, n_mfcc=13, fs=16000):

    mfcc = librosa.feature.mfcc(input_wav, sr=fs, hop_length=hop_length, win_length=frame_size, n_mfcc=n_mfcc)

    return mfcc


def serialize_spectrogram(data, serialization_window_ms, serialization_hop_size_ms, frame_hop_size_ms, fs=16000):

    slices, start_ms, end_ms = [], [], []

    total_num_frames = data.shape[-1]

    frame_hop_size = ms_to_samples(frame_hop_size_ms, fs)

    total_audio_ms = frames_to_ms(total_num_frames, frame_hop_size, fs)
    print('Waveform is {}ms duration'.format(total_audio_ms))

    frame_len_ms = frames_to_ms(1, frame_hop_size, fs)

    assert serialization_window_ms % frame_len_ms == 0
    assert serialization_hop_size_ms % frame_len_ms == 0

    num_window_frames = int(serialization_window_ms / frame_len_ms)
    num_hop_frames = int(serialization_hop_size_ms / frame_len_ms)

    if num_window_frames > total_num_frames:
        slices.append(process_incomplete_slice(data, num_window_frames))
        start_ms.append(0)
        end_ms.append(frames_to_ms(num_window_frames, frame_hop_size, fs))
    else:
        for end_idx in range(num_window_frames, total_num_frames, num_hop_frames):
            start_idx = end_idx - num_window_frames
            slices.append(data[:,start_idx:end_idx])
            start_ms.append(frames_to_ms(start_idx, frame_hop_size, fs))
            end_ms.append(frames_to_ms(end_idx, frame_hop_size, fs))

    num_leftover_frames = (total_num_frames - num_window_frames) % num_hop_frames
    num_leftover_ms = frames_to_ms(num_leftover_frames, frame_hop_size, fs)

    return slices, start_ms, end_ms


def frames_to_ms(num_frames, feat_hop_size, fs):
    return int(np.floor(1000 * num_frames * feat_hop_size / fs))


def ms_to_samples(ms, fs):
    return int(ms * fs / 1000)


def process_incomplete_slice(data, num_window_frames, option='eps-pad-start'):


    total_num_frames = int(data.shape[-1])
    num_missing_frames = num_window_frames - total_num_frames

    output_slice = np.empty((data.shape[0], num_window_frames), dtype=np.float32)


    if option == 'eps-pad-start':
        for t in range(num_window_frames):
            if t < num_missing_frames:
                eps = np.random.uniform(0, 1e-6, None)
                output_slice[:,t].fill(eps)
            else:
                output_slice[:,t] = data[:,t]
    else:
        # nothing else supported yet
        raise Exception

    return output_slice

if __name__ == '__main__':
    
    # file obtained from https://www.kaggle.com/aanhari/alexa-dataset
    flist = ['data/alexa/alexa/anfcucvo/1.wav', 'data/alexa/alexa/anfcucvo/2.wav', 'data/alexa/alexa/anfcucvo/3.wav']
    serialize_spectrograms_from_wav_files(input_flist=flist, fs=16000)
