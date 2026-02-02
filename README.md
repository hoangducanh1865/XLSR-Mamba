# XLSR-Mamba Official Pytorch Implementation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/xlsr-mamba-a-dual-column-bidirectional-state/audio-deepfake-detection-on-asvspoof-2021)](https://paperswithcode.com/sota/audio-deepfake-detection-on-asvspoof-2021?p=xlsr-mamba-a-dual-column-bidirectional-state)

Official Implementation of our work ["XLSR-Mamba: A Dual-Column Bidirectional State Space Model for Spoofing Attack Detection"](https://arxiv.org/pdf/2411.10027) published in IEEE Signal Processing Letters. For detailed insights into our methodology, you can access the complete paper.
If you have any questions on this repository or the related paper, please [create an issue](https://github.com/swagshaw/XLSR-Mamba/issues/new) or [email me](mailto:yxiao009+github@e.ntu.edu.sg).

## News
[Mar. 2025] We would like to inform you that the scores for XLSR-Mamba achieves 2nd Rank on the [Speech-DF-Arena leaderboard](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena). 

[Feb. 2025] We release code, checkpoints, and extracted scores for XLSR-Mamba.

## Getting Started
### Setup Environment
You need to create the running environment by [Anaconda](https://www.anaconda.com/).
First, create and activate the environment:

```bash
conda create -n XLSR_Mamba python=3.10
conda activate XLSR_Mamba
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

Install fairseq:

```bash
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
```
### Pretrained XLSR
The pretrained model XLSR can be found at this [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

### Datasets
We used the LA partition of ASVspoof 2019 for training and validation, it can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).

We used the ASVspoof 2021 database for evaluation. LA can be found [here](https://zenodo.org/records/4837263#.YnDIinYzZhE), and DF [here](https://zenodo.org/records/4835108#.YnDIb3YzZhE).

We also used the 'In-the-Wild' dataset for evaluation, it can be downloaded from [here](https://deepfake-total.com/in_the_wild).
## Training & Testing on fixed-length input
To train and produce the score for the LA set evaluation, run:
```bash
python main.py --algo 5
```
To train and produce the score for DF set evaluation, run:
```bash
python main.py --algo 3
```
Upon running this command, a new folder will be generated within the 'models' directory, containing the top 5 epochs from this training loop. Additionally, two score files will be created: one for LA and another for DF, both located in the 'Scores' folder.
You can evaluate the score by:
```bash
bash evaluate.sh
```
Please remember to choose the correct score file and the dataset path.

### Pretrained models
We have uploaded the pretrained models of our experiments to 🤗 HugginFace:
<div align="center">

| **Model** | **#Total Params** | **Download** |
| :------------: | :------------: | :------------: |
| XLSR-Mamba-LA | 319M  | [🤗 HuggingFace](https://huggingface.co/AustinXiao/XLSR-Mamba-LA)  |
| XLSR-Mamba-DF | 319M  | [🤗 HuggingFace](https://huggingface.co/AustinXiao/XLSR-Mamba-DF)  |

</div>

## Results

<div align="center">
  
|Dataset | **EER (%)** | **min t-DCF** |
|--------|--------------------|--------------------|
|ASVspoof2021 LA  | **0.93**  | **0.208**          |
|ASVspoof2021 DF  | **1.88**  | **-**              | 
|In-The-Wild      | **6.71**  | **-**              |

</div>

## Citation
If you find our repository valuable for your work, please consider giving a star to this repo and citing our paper:
```
@article{xiao2024xlsr,
  title={{XLSR-Mamba}: A Dual-Column Bidirectional State Space Model for Spoofing Attack Detection},
  author={Xiao, Yang and Das, Rohan Kumar},
  journal={arXiv preprint arXiv:2411.10027},
  year={2024}
}
```

## Acknowledgements
Our implementations use the source code from the following repositories and users:

- [conformer-based-classifier-for-anti-spoofing](https://github.com/ErosRos/conformer-based-classifier-for-anti-spoofing)
- [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
- [tcm_add](https://github.com/ductuantruong/tcm_add)

## License
The project is available as open source under the terms of the [MIT License](./LICENSE).
