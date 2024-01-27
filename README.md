# FastPhonemicUniversalAdversarialAttack

## Introdction

Fast**Phonemic**Universal**Adversarial****Attack**

## Requirements

* PyTorch 1.10.2
* Python 3.6

```
pip install -r requirements.txt
```
## Usage

First, you should download model parameter files of [DeepSpeech2](https://github.com/SeanNaren/deepspeech.pytorch). and put it in model/model_pth.

Second, you should download [LibriSpeech](https://www.openslr.org/12)
to test the framework.

We provide verification dataset in the source folder, which you can run directly when the model and data are ready.
```
python FPUAA.py
```






