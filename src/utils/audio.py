import os, sys
sys.path.append('./venv/lib/python3.9/site-packages')
import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from src.utils.hparams import hparams as hp
import soundfile as sf
import pandas as pd

def load_wav(path: object, sr: object) -> object:
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)


def generate_silence(duration, sample_rate=44100):
    # 计算生成音频所需的采样点数
    num_samples = int(duration * sample_rate)
    # 生成全零的音频数据
    audio_data = np.zeros((num_samples, 1), dtype=np.float32)
    return audio_data, sample_rate

import wave


def merge_wav_files(input_files, output_file):
    # 打开输出文件
    with wave.open(output_file, 'wb') as output_wav:
        # 设置输出文件的参数（与输入文件的参数一致）
        with wave.open(input_files[0], 'rb') as first_wav:
            output_wav.setparams(first_wav.getparams())

        # 依次写入每个输入文件的音频数据
        for input_file in input_files:
            with wave.open(input_file, 'rb') as input_wav:
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))

def get_wav_duration(file_path):
    # 打开 WAV 文件
    with wave.open(file_path, 'rb') as wav_file:
        # 获取总帧数
        num_frames = wav_file.getnframes()
        # 获取采样频率
        sample_rate = wav_file.getframerate()
        # 计算时长（以秒为单位）
        duration = num_frames / float(sample_rate)
        return duration

def get_audios_duration_and_save(audio_paths, output_path):
    audio_names = []
    audio_durations = []
    for audio_path in audio_paths:
        duration = get_wav_duration(audio_path)
        audio_with_extension = os.path.basename(audio_path)
        audio_name_without_extension = os.path.splitext(audio_with_extension)[0]
        audio_names.append(audio_name_without_extension)
        audio_durations.append(duration)
    dataFrame = pd.DataFrame({'audio_name': audio_names, 'duration': audio_durations})
    dataFrame.to_csv(output_path, index=False, sep=',')

def readTextWithExcel(csv_path):
    df = pd.read_excel('0330.xlsx')
    # print(df.iloc[2, 1])
    dic = {}
    for i in range(0, 4):
        print(str(i) + str(df.iloc[i, 0]) + ':' + df.iloc[i, 1])
        if df.iloc[i, 0] in dic:
            dic[df.iloc[i, 0]] = dic[df.iloc[i, 0]] + df.iloc[i, 1]
        else:
            dic[df.iloc[i, 0]] = df.iloc[i, 1]
    print(dic)
    return dic

if __name__ == '__main__':
    # 生成静音音频数据
    # duration = 5  # 时长（秒）
    # silence_data, sample_rate = generate_silence(duration)

    # 保存静音音频文件
    # output_path = 'silence_audio.wav'
    # sf.write(output_path, silence_data, sample_rate)
    #
    # g_duration = get_wav_duration(output_path)

    # 输入要合并的音频文件列表

    input_files = ['1.wav', '2.wav', '3.wav', '4.wav']
    input_paths = []

    for input_file in input_files:
        input_paths.append(os.path.join('./output6/', input_file))

    # 指定输出文件名
    output_file = 'output.wav'

    #get_audios_duration_and_save(input_paths, 'audio_durations_output.csv')

    # 调用函数进行合并
    #merge_wav_files(input_paths, output_file)

    data = pd.read_csv('audio_durations_output.csv', sep=',')
    for duration in data['duration']:
        print(duration)

    print(data)


    # print(f"Silence audio file saved successfully. duration: {g_duration}")
