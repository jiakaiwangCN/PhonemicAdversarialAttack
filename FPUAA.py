import datetime
import random

import torchaudio
import torch
from torch.utils import data
import editdistance
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from g2p_en import G2p
import torch.nn.functional as F

from model.pytorch_model import PyTorchAudioModel
from model.utils import load_decoder, load_model

# load model
model_path = '...'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(model_path=model_path)
decoder = load_decoder(labels=model.labels)
audio_model = PyTorchAudioModel(model, decoder, device)


# compute cer
def compute_cer(origin, noise):
    return editdistance.eval(origin, noise) / len(origin)


def superpose(x, v):
    x = x.to(device)
    x_len = x.shape[1]
    v_len = v.shape[1]
    start = random.randint(0, v_len)
    while v_len < x_len:
        v = torch.cat((v, v), dim=1)
        v_len = v.shape[1]
    v = torch.cat((v, v), dim=1)
    v = v[:, start:x_len + start]
    v = v.to(device)
    res = x + v
    res = torch.clamp(res, -1, 1)
    return res, start


# compute sr and cer
def sr_and_cer(sound_data, noise, model):
    i = 0
    cer = 0.0
    sound_len = len(sound_data)
    for sound in sound_data:
        new_sound, start = superpose(sound['wave'], noise)
        predict = model(new_sound, decode=True)[0][0][0]
        pred_cer = compute_cer(sound['label'], predict)
        cer += pred_cer
        if pred_cer > 0.5:
            i += 1
    print('SR = ' + str(i / sound_len))
    print('CER = ' + str(cer / sound_len))


# load rir audio data
def get_rir_sample(path, processed=False):
    rir_raw, sample_rate = torchaudio.load(path)
    if rir_raw.shape[0] > 1:
        rir_raw = rir_raw[0].unsqueeze(0)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate * 1.01):int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate


# transform audio by rir audio
def rir_transform(waveform, sample_rate, rir_path):
    rir_raw, _ = get_rir_sample(path=rir_path)
    rir = rir_raw / torch.norm(rir_raw, p=2)
    rir = torch.flip(rir, [1])
    rir = rir.to(device)
    speech_ = F.pad(waveform, (rir.shape[1] - 1, 0))
    augmented = F.conv1d(speech_[None, ...], rir[None, ...])[0]
    return augmented


# load rir audio list
def generate_rir_list():
    rir_list = []
    source = 'rir_list'
    for rir in os.listdir(source):
        if os.path.splitext(rir)[1] == ".wav":
            side = os.path.join(source, rir)
            rir_list.append(side)
    return rir_list


# compute mean phoneme length and mean phoneme density
def compute_mean_phoneme_length():
    g2p = G2p()
    libri_speech_dataset = torchaudio.datasets.LIBRISPEECH("...", url='dev-clean', download=False)
    libri_speech_loader = data.DataLoader(dataset=libri_speech_dataset, batch_size=1, shuffle=False)
    index = 0
    sum = 0
    for i, y in enumerate(tqdm(libri_speech_loader)):
        sound = y[0][0]
        label = y[2][0]
        phoneme_list = g2p(label)
        s_num = len(phoneme_list)
        s_len = sound.shape[1] / 16000
        sum += s_len / s_num
        index += 1
    mean_phoneme_length = sum / index
    print('mean phoneme length is ' + str(mean_phoneme_length))
    mean_phoneme_density = 1 / mean_phoneme_length
    print('mean phoneme density is ' + str(mean_phoneme_density))


# show scatter of this dataset
def audio_scatter():
    g2p = G2p()
    libri_speech_dataset = torchaudio.datasets.LIBRISPEECH("...", url='dev-clean', download=False)
    libri_speech_loader = data.DataLoader(dataset=libri_speech_dataset, batch_size=1, shuffle=False)
    len_list = []
    density_list = []
    for i, y in enumerate(tqdm(libri_speech_loader)):
        sound = y[0][0]
        label = y[2][0]
        phoneme_list = g2p(label)
        s_num = len(phoneme_list)
        s_len = sound.shape[1] / 16000
        density = s_num / s_len
        len_list.append(s_len)
        density_list.append(density)
    plt.scatter(density_list, len_list, s=2)
    plt.xlabel('phoneme density')
    plt.ylabel('audio duration (s)')
    plt.show()


# select train audio list
def generate_data(num, alpha, is_shuffle=False):
    g2p = G2p()
    dev_url = "dev-clean"
    pipeline_dataset = torchaudio.datasets.LIBRISPEECH("...", url=dev_url, download=False)
    pipeline_loader = data.DataLoader(dataset=pipeline_dataset,
                                      batch_size=1,
                                      shuffle=is_shuffle)
    train_data = []
    index = 0
    for i, y in enumerate(pipeline_loader):
        sound = y[0][0]
        label = y[2][0]
        phoneme_list = g2p(label)
        s_num = len(phoneme_list)
        s_len = sound.shape[1] / 16000
        density = s_num / s_len
        if sound.shape[1] < 213360 or density < 12.4 - alpha or density > 12.4 + alpha \
                or torch.max(sound) < 0.4727 or torch.max(sound) > 0.6991:
            continue
        if index < num:
            train_data.append({"wave": sound, "label": label})
        else:
            break
        index += 1
    return train_data


# main framework
class FPUAA:

    # init
    def __init__(self):
        super(FPUAA, self).__init__()
        self.criterion = nn.CTCLoss()
        self.softmax = nn.Softmax(dim=2)

    # word -> int vector
    def target_sentence_to_label(self, sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
        out = []
        for word in sentence:
            out.append(labels.index(word))
        return torch.IntTensor(out)

    # superpose sound and noise from start
    def superpose(self, sound, noise, start):
        sound = sound.to(device)
        sound_len = sound.shape[1]
        noise_len = noise.shape[1]
        while noise_len < sound_len:
            noise = torch.cat((noise, noise), dim=1)
            noise_len = noise.shape[1]
        noise = torch.cat((noise, noise), dim=1)
        noise = noise[:, start:sound_len + start]
        noise = noise.to(device)
        res = sound + noise
        res = torch.clamp(res, -1, 1)
        return res, start

    # main method
    def generate(self, train_data, epoch_size, model, device, adv_len, eps,
                 step, iteration=100, lr=1e-1, rir_list=None):
        noise_init = train_data[0]['wave'][:, :adv_len]
        noise = torch.FloatTensor(noise_init).view(1, -1).to(device).requires_grad_(True)
        noise.data.clamp_(min=-eps, max=eps)
        data_length = len(train_data)
        rand_rir = None
        rand_rir_idx = None
        for epoch in range(epoch_size):
            for index, sound in tqdm(enumerate(train_data)):
                one_sound = sound['wave']
                one_sound = one_sound.to(device)
                label = sound['label']
                targets = self.target_sentence_to_label(label)
                targets = targets.view(1, -1).to(device).detach()
                target_lengths = torch.IntTensor([targets.shape[1]]).view(1, -1)
                optimizer = torch.optim.Adam([noise], lr=lr)
                model.zero_grad()
                step_start = 0
                for cur_index in range(iteration):
                    optimizer.zero_grad()
                    advs, start = self.superpose(one_sound, noise, step_start)
                    step_start += step
                    if step_start > adv_len:
                        step_start -= adv_len
                    advs = advs.to(device)
                    if rir_list is not None and step_start < step:
                        rand_rir_idx = np.random.randint(len(rir_list))
                        rand_rir = rir_list[rand_rir_idx]
                    if rand_rir is not None:
                        advs = rir_transform(advs, 16000, rand_rir)
                        advs.data.clamp_(min=-1, max=1)
                    decode_out, out, output_sizes = model(advs, decode=True)
                    out = self.softmax(out)
                    out = out.transpose(0, 1).log()
                    loss_CTC = self.criterion(out, targets, output_sizes, target_lengths)
                    loss = - loss_CTC
                    loss.backward()
                    noise.grad.nan_to_num_(nan=0)
                    optimizer.step()
                    noise.data.clamp_(min=-eps, max=eps)
                    cer = compute_cer(label, decode_out[0][0])
                    print(
                        "epoch = {}/{}; data = {}/{}; iter = {}/{}; loss = {:.2f}; rir = {};"
                        " cur_cer = {:.4f}; start = {}; decode_out = {}".format(epoch + 1, epoch_size,
                                                                                index + 1, data_length,
                                                                                cur_index + 1,
                                                                                iteration, loss.item(),
                                                                                rand_rir_idx, cer, start,
                                                                                decode_out[0][0]))
        return noise.detach()


if __name__ == '__main__':
    attacker = FPUAA()
    test_data = np.load('source/100_test_audio_list.npy', allow_pickle=True)
    train_data = generate_data(10, 0.2, True)
    rir_list = generate_rir_list()
    start = datetime.datetime.now()
    noise = attacker.generate(train_data, 3, audio_model, device, adv_len=3200,
                              eps=0.01, step=2467, iteration=30, lr=2e-3)
    end = datetime.datetime.now()
    time = end - start
    print('Time = ' + str(end - start))
    sr_and_cer(test_data, noise, audio_model)
