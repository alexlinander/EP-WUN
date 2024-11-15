# EP-WUN

## Dataset prepare
For the implementation, please first prepare 12 files under `./dataset`, and put training , testing, and validation set in it in the form of `clean_trainset_8k`, `clean_trainset_16k`, `noisy_trainset_8k`, `noisy_trainset_16k`, `clean_validset_8k`, `clean_validset_16k`, `noisy_validset_8k`, `noisy_validset_16k`, `clean_testset_8k`, `clean_testset_16k`, `noisy_testset_8k`, `noisy_testset_16k` respectively.

It should be noted that each audio file in the dataset should be segmented into 1.024 second intervals, and that the "noisy" set should be aligned with the "clean" set in terms of speech content.


## Training
To train EP-WUN, first navigate to the `./train` folder and the training process consists of 3 steps, which are:
1. Train the Wave-U-Net
```
python train_waveunet.py
```
2. After obtaining the bandwidth expansion based Wave-U-Net, train the SQC
```
python train_SQC.py
```
3. After the Wave-U-Net and SQC are both prepared, train the EP-WUN
```
python train_EPWUN.py
```

## Citation
```
@inproceedings{lin23f_interspeech,
  title     = {Noise-Robust Bandwidth Expansion for 8K Speech Recordings},
  author    = {Yin-Tse Lin and Bo-Hao Su and Chi-Han Lin and Shih-Chan Kuo and Jyh-Shing Roger Jang and Chi-Chun Lee},
  year      = {2023},
  booktitle = {INTERSPEECH 2023},
  pages     = {5107--5111},
  doi       = {10.21437/Interspeech.2023-857},
  issn      = {2958-1796},
}
```

