
from defense.speech_compression import *

if __name__ == '__main__':

    print('testing the installation of ffmpeg and the en-/de-coders')

    '''
    testing the installation of ffmpeg and the en-/de-coders
    '''
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('audio_path')

    args = parser.parse_args()

    import torchaudio
    audio, _ = torchaudio.load(args.audio_path)

    for f in [OPUS, SPEEX, AMR, AAC_V, AAC_C, MP3_V, MP3_C]:
        f_audio = f(audio, debug=False)
    print('*' * 50, 'Speech compression is ready. Enjoy It.', '*' * 50)