import numpy as np
import scipy.io.wavfile as scw
import sys
import scipy

win_length = 2048
overlap = 4

def remove_zeros(audio):
    i= 0
    while np.sum(np.abs(audio[i:i+10])) < 60:
        i += 1
    print (i)
    return audio[i:]

def stft(x, fftsize, overlap):
    """Returns short time fourier transform of a signal x
    """
    hop = int(fftsize / overlap)
    w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0,len(x)-fftsize, hop)])

def istft(X, overlap):
    """Returns inverse short time fourier transform of a complex spectrum X
    """
    fftsize=(X.shape[1]-1)*2
    hop = int(fftsize / overlap)
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop)
    for n,i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   #overlap-add
        wsum[i:i+fftsize] += w ** 2.
    return x

def signalabs(arr):
    return (scipy.absolute(arr), scipy.angle(arr))


if __name__ == "__main__":
    orig_path = sys.argv[1]
    instr_path = sys.argv[2]
    fn = sys.argv[3]

    sampling_rate, orig_raw = scw.read(orig_path)
    instr_raw = scw.read(instr_path)[1]

    orig_audio = remove_zeros(orig_raw)
    instr_audio = remove_zeros(instr_raw)

    if len(orig_audio) > len(instr_audio):
        orig_audio = orig_audio[0:len(instr_audio)]
    elif len(orig_audio) < len(instr_audio):
        instr_audio = instr_audio[0:len(orig_audio)]

    o_windows_l = stft(orig_audio[:,0], win_length, overlap)
    o_windows_r = stft(orig_audio[:,1], win_length, overlap)
    i_windows_l = scipy.absolute(stft(instr_audio[:,0], win_length, overlap))
    i_windows_r = scipy.absolute(stft(instr_audio[:,1], win_length, overlap))

    o_vals_l = signalabs(o_windows_l)
    o_windows_l = o_vals_l[0]
    o_imags_l = o_vals_l[1]

    o_vals_r = signalabs(o_windows_r)
    o_windows_r = o_vals_r[0]
    o_imags_r = o_vals_r[1]

    o_windows_l = np.clip(o_windows_l - (1 * i_windows_l), 0, np.amax(o_windows_l))
    o_windows_r = np.clip(o_windows_r - (1 * i_windows_r), 0, np.amax(o_windows_r))

    final_windows_l = istft(o_windows_l * scipy.exp(o_imags_l*1j), overlap)
    final_windows_r = istft(o_windows_r * scipy.exp(o_imags_r*1j), overlap)

    filtered = np.stack((final_windows_l, final_windows_r), axis=1)

    vol_factor = 20000/np.amax(np.absolute(filtered))
    filtered = filtered * vol_factor

    scw.write("test.wav", sampling_rate, filtered.astype(np.int16))
