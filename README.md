# AVSA

This repository contains the code and dataset in our NeurIPS'20 paper.

[Learning Representations from Audio-Visual Spatial Alignment](https://papers.nips.cc/paper/2020/file/328e5d4c166bb340b314d457a208dc83-Paper.pdf).
Pedro Morgado*, Yi Li*, Nuno Vasconcelos.
*Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

## Prerequisites

Requirements listed in `environment.yml`.

## Data preparation

### [YT-360 dataset](https://pedro-morgado.github.io/AVSpatialAlignment/)

YouTube id's of videos in the YT-360 dataset are provided in `datasets/assets/yt360/[train|test].txt`, and segment timestamps in `datasets/assets/yt360/segments.txt`.
Please, use your favorite YouTube dataset downloader to download the videos (e.g.~[link](https://github.com/rocksyne/kinetics-dataset-downloader)), and split them into 10s clips. 
The dataset should be stored in `data/yt360/video` and `data/yt360/audio` with filenames `{YOUTUBE_ID}-{SEGMENT_START_TIME}.{EXTENSION}`.

The pre-extracted segmentation maps can be downloaded from [here](https://nextcloud.nrp-nautilus.io/s/zYisGXab9EJPtFB) and extracted to `data/yt360/segmentation/`. 

If you experience issues downloading or processing the dataset, please email the authors at {[pmaravil](mailto:pmaravil@eng.ucsd.edu), [yil898](mailto:yil898@eng.ucsd.edu)}@eng.ucsd.edu for assistance.

## Pre-trained model
The AVSA model that yield the top performance (trained from `configs/main/avsa/Cur-Loc4-TransfD2.yaml`) is available [here](https://nextcloud.nrp-nautilus.io/s/T9SD8xn2pCHHCKG).

## Self-supervised training

```
python main-video-ssl.py [--quiet] cfg
```

Training config `cfg` for the following models are provided:
- AVC training (instance discrimination): `configs/main/avsa/InstDisc.yaml`
- AVSA training: `configs/main/avsa/Cur-Loc4-TransfD2.yaml`
- AVSA training w/o curriculum: `configs/main/avsa/NoCur-Loc4-TransfD2.yaml`

## Evaluation

Four downstream tasks are supported: Binary audio-visual correspondence (AVC-Bin), binary audio-visual spatial alignment (AVSA-Bin), video action recognition (on UCF/HMDB), and audio-visual semantic segmentation.

### Action recognition

```
python eval-action-recg.py [--quiet] cfg model_cfg
```

Evaluation config `cfg` for UCF and HMDB dataset are provided:
- UCF: `configs/benchmark/ucf/ucf-8at16-fold[1|2|3].yaml`
- HMDB: `configs/benchmark/hmdb/hmdb-8at16-fold[1|2|3].yaml`

`model_cfg` is training config for the model to evaluate, e.g. `configs/main/avsa/Cur-Loc4-TransfD2.yaml` for AVSA pre-training.

### Semantic segmentation

```
python eval-audiovisual-segm.py [--quiet] cfg model_cfg
```

Evaluation config `cfg` for three settings are provided:
- Visual segmentation: `configs/benchmark/segmentation/yt360-fpn-4crop-head-vonly.yaml`
- Visual+audio segmentation: `configs/benchmark/segmentation/yt360-fpn-4crop-head-audio.yaml`
- Visual+audio segmentation with context: `configs/benchmark/segmentation/yt360-fpn-4crop-head-audio-ctx.yaml`

### Binary audio-visual correspondence

```
python eval-avc.py [--quiet] cfg model_cfg
```

Evaluation config `cfg` for two settings are provided:
- With transformer: `configs/benchmark/avc/avc-transf-[1|4]crop.yaml`
- Without transformer: `configs/benchmark/avc/avc-notransf-[1|4]crop.yaml`

### Binary audio-visual spatial alignment

```
python eval-avsa.py [--quiet] cfg model_cfg
```

Evaluation config `cfg` for two settings are provided:
- With transformer: `configs/benchmark/avsa/avsa-transf-[1|4]crop.yaml`
- Without transformer: `configs/benchmark/avsa/avsa-notransf-[1|4]crop.yaml`

## Citations

Please cite our work if you find it helpful for your research:

```
@article{morgado2020learning,
  title={Learning Representations from Audio-Visual Spatial Alignment},
  author={Morgado, Pedro and Li, Yi and Nvasconcelos, Nuno},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Acknowledgements

This work was partially funded by NSF award IIS-1924937 and NVIDIA GPU donations. We also acknowledge and thank the use of the Nautilus platform for some of the experiments in paper.
