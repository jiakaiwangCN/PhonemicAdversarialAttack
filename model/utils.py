import torch
from model.deepspeech.inference_config import TranscribeConfig
from model.deepspeech.decoder import BeamCTCDecoder, GreedyDecoder
from model.deepspeech.model import DeepSpeech
from model.deepspeech.enums import DecoderType
import editdistance
import torchaudio.transforms as T


def load_model(model_path):
    info = torch.load(model_path, 'cpu')
    hyper_parameters = info['hyper_parameters']
    labels = hyper_parameters['labels']
    precision = hyper_parameters['precision']
    model = DeepSpeech(labels, precision)
    model.load_state_dict(info['state_dict'])
    # model.eval()
    model.train()
    return model


def load_decoder(labels):
    cfg = TranscribeConfig
    lm = cfg.lm
    if lm.decoder_type == DecoderType.beam:
        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=lm.lm_path,
                                 alpha=lm.alpha,
                                 beta=lm.beta,
                                 cutoff_top_n=lm.cutoff_top_n,
                                 cutoff_prob=lm.cutoff_prob,
                                 beam_width=lm.beam_width,
                                 num_processes=lm.lm_workers,
                                 blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels=labels,
                                blank_index=labels.index('_'))
    return decoder


def distance(y, t, blank='<eps>'):
    def remap(y, blank):
        prev = blank
        seq = []
        for i in y:
            if i != blank and i != prev: seq.append(i)
            prev = i
        return seq

    y = remap(y, blank)
    t = remap(t, blank)
    return y, t, editdistance.eval(y, t)


# wrong
def calculate_wer(y, label):
    total_pred_wer, total_word = 0, 0
    pred = [i for i in y]
    lab = [i for i in label]

    pred1, lab, pred_wer = distance(pred, lab)
    total_pred_wer += pred_wer
    total_word += len(lab)
    return 100 * total_pred_wer / total_word


# wrong
def calculate_cer(y, t):
    y = ' '.join(y)
    t = ' '.join(t)
    char_t_len = len(t)
    return char_t_len, editdistance.eval(list(y), list(t))


def cal_cer(word1: str, word2: str) -> float:
    # word1 : predict
    # word2 : label

    row = len(word1) + 1
    column = len(word2) + 1

    cache = [[0] * column for i in range(row)]

    for i in range(row):
        for j in range(column):

            if i == 0 and j == 0:
                cache[i][j] = 0
            elif i == 0 and j != 0:
                cache[i][j] = j
            elif j == 0 and i != 0:
                cache[i][j] = i
            else:
                if word1[i - 1] == word2[j - 1]:
                    cache[i][j] = cache[i - 1][j - 1]
                else:
                    replace = cache[i - 1][j - 1] + 1
                    insert = cache[i][j - 1] + 1
                    remove = cache[i - 1][j] + 1

                    cache[i][j] = min(replace, insert, remove)

    return cache[row - 1][column - 1]/len(word2)


def cal_wer(word1: str, word2: str) -> float:
    # word1 : predict
    # word2 : label

    word1 = word1.split(' ')
    word2 = word2.split(' ')

    row = len(word1) + 1
    column = len(word2) + 1

    cache = [[0] * column for i in range(row)]

    for i in range(row):
        for j in range(column):

            if i == 0 and j == 0:
                cache[i][j] = 0
            elif i == 0 and j != 0:
                cache[i][j] = j
            elif j == 0 and i != 0:
                cache[i][j] = i
            else:
                if word1[i - 1] == word2[j - 1]:
                    cache[i][j] = cache[i - 1][j - 1]
                else:
                    replace = cache[i - 1][j - 1] + 1
                    insert = cache[i][j - 1] + 1
                    remove = cache[i - 1][j] + 1

                    cache[i][j] = min(replace, insert, remove)

    return cache[row - 1][column - 1] / len(word2)


sample_rate = 16000
n_fft = int(sample_rate * 0.02)
hop_length = int(sample_rate * 0.01)
win_length = int(sample_rate * 0.02)
# n_fft = 1024
# win_length = 320 #20ms
# hop_length = 160 #10ms
n_mels = 80
n_mfcc = 80  # 23
mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
                        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length,
                                   'hop_length': hop_length})


def data_processing(data, data_type="test"):
    mfcc = []
    label = []
    fileid_audio = " "
    for (waveform, _, utterance, speaker_id, chapter_id, utterance_id) in data:
        if data_type == 'test':
            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            label = utterance.lower().split()
            fileid_audio = str(speaker_id) + "-" + str(chapter_id) + "-" + str(utterance_id)

    return mfcc, label, fileid_audio
