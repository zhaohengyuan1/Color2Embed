# Color2Embed
Color2Embed: Fast Exemplar-Based Image Colorization using Color Embeddings [Paper](https://arxiv.org/abs/2106.08017)

This project is the simple implementation of Color2Embed. This paper maybe not be submitted to any conferences and journals and you can use it in your projects. You can help to star this repo if you think this repo is helpful.

<p align="left">
  <img src="./misc/fig1.png">
</p>

Other recommended projects:

[Temporally-Consistent-Video-Colorization](https://github.com/lyh-18/TCVC-Temporally-Consistent-Video-Colorization)

[Deep Exemplar-based Colorization](https://github.com/msracver/Deep-Exemplar-based-Colorization)

[Deep Video Prior](https://github.com/ChenyangLEI/deep-video-prior)

[Learning Blind Video Temporal Consistency](https://github.com/phoenix104104/fast_blind_video_consistency)

[Swapping Autoencoder](https://github.com/taesungp/swapping-autoencoder-pytorch)

## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.5.0](https://pytorch.org/)

## Test
1. Clone this github repo. 
```
git clone https://github.com/zhaohengyuan1/Color2Embed.git
cd Color2Embed
```

2. [Pretrained models](https://drive.google.com/file/d/15qgn3aSzviBE4tW6PaCx4c_syiKMBVir/view?usp=sharing) should be placed in `./experiments/` folder. [VGG model](https://drive.google.com/file/d/1eMiUDeO_YGOu3RfyQKeAkKR5rgvH5a3d/view?usp=sharing) also can be downloaded.

3. I have collected some test datasets used in previous papers. You can check it in the path `./test_datasets`. When you use the file `test_gray2color.py` to test, you need to edit the input file path and pretrained weights path in this file.

```
python test_gray2color.py
```

## Train

1. Prepare the training data.

```
cd data
sh prepare_data.sh
```

2. Run the train.sh. You can check `train.py` for more implementation details.

```
sh train.sh
```



## Results

<p align="left">
  <img  src="./misc/fig3.png">
</p>

<p align="left">
  <img src="./misc/fig4.png">
</p>


## Contact
If you have any question, please email hubylidayuan@gmail.com.
