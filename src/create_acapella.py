import numpy as np
import scipy.io.wavfile as scw

winLength = 2048
overlap = 4


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



if __name__ == "__main__":
    orig_path = sys.argv[1]
    instr_path = sys.argv[2]
    fn = sys.argv[3]
