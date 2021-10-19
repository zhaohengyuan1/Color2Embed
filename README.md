# Color2Embed
Color2Embed: Fast Exemplar-Based Image Colorization using Color Embeddings [Paper](https://arxiv.org/abs/2106.08017)

This project is the simple implementation of Color2Embed. You can use it in your own projects. You can help to star this repo if you think this repo is helpful.

<p align="left">
  <img src="./misc/fig1.png">
</p>

## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.5.0](https://pytorch.org/)

## Test
1. Clone this github repo. 
```
git clone https://github.com/zhaohengyuan1/Color2Embed.git
cd Color2Embed
```

2. Pretrained models should be Downloaded in `./experiments/` folder. The model will be uploaded.

3. I have collected some test datasets used in previous papers. You can check it in the path `./test_datasets`. When you use the file `test_gray2color.py` to test, you need to edit the input file path and pretrained weights path in this file.

```
python test_gray2color.py
```

## Train

1. Prepare the training data

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

If you find our work is useful, please kindly cite it.
```
@misc{zhao2021color2embed,
      title={Color2Embed: Fast Exemplar-Based Image Colorization using Color Embeddings}, 
      author={Hengyuan Zhao and Wenhao Wu and Yihao Liu and Dongliang He},
      year={2021},
      eprint={2106.08017},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
If you have any question, please email hubylidayuan@gmail.com.