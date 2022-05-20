import sounddevice
from scipy.io.wavfile import write

def record(filename):
    sr = 16000
    sec = 4
    print("Recording")

    rec = sounddevice.rec(int(sec * sr), samplerate=sr, channels=1)
    sounddevice.wait()
    write(filename + ".wav", sr, rec)