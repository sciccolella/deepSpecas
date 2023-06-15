"""
python poly/esmPredict.py --csv  /home/simone/data/benchmark/2samples/1/predict.merged.csv --pyplot2s /home/simone/polyDS/net/mincov3/pyplot2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth --colmplp2s /home/simone/polyDS/net/mincov3/colmplp2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth --squishstack /home/simone/polyDS/net/mincov3/squishstack/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth --squish4d /home/simone/polyDS/net/mincov3/squish4d/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth --classes A3 A5 CSE RI NN
"""

import torch

# import torchvision.transforms as transforms
# from torchvision.models import resnet50, ResNet50_Weights
import sys
import torch.nn as nn

# from torch.utils.data import random_split as pytorch_randsplit

# # from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
# # from torchvision.models import inception_v3, Inception_V3_Weights
# # import torch.optim as optim
# # from torchvision import datasets
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from matplotlib import pyplot as plt
# from PIL import Image
import torch.nn.functional as F

# import copy
# import time
# from rich.progress import track
# from torch.utils.data import Dataset
# import pandas as pd
from os.path import join as pjoin
from os import makedirs
import pysam

# from torch import topk
# import cv2

# from os import makedirs

# from torchvision.io import read_image

from functools import partial

import deepSpecas
from plot2s import Event2sample, Event2sampleReads, Event2sampleComb, parse_region


import time

def predict(
    csvpath: str,
    models: list,
    plots: list,
    transformers: list,
    mincov: int,
    classes: list[str],
    imgout: str = "",
):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"

    # TODO: this can be optimized using batches and vmap
    # https://pytorch.org/functorch/stable/notebooks/ensembling.html

    nets = [m.to(device) for m in models]
    MAXREADS=5_000

    _event_ix = 0
    print("ix,region,pred", sep=",")
    with torch.no_grad():
        for line in open(csvpath):
            _event_ix += 1
            line = line.strip()
            region, bam1path, bam2path = line.split(",")
            print(_event_ix, region, sep=",", end=",")
            outs = []

            for plt_ix, _ in enumerate(plots):

                _cls = Event2sample
                if plots[plt_ix] in ["squishstack", "squish4d"]:
                    _cls = partial(Event2sampleReads, max_reads=MAXREADS)
                elif plots[plt_ix] in ["comb", "comb4d"]:
                    _cls = partial(Event2sampleComb, max_reads=MAXREADS)

                event = _cls(parse_region(region), pysam.AlignmentFile(bam1path), pysam.AlignmentFile(bam2path))
                _p = event.plot(zoom=args.zoom, type=plots[plt_ix], mode="pil", prefix='', crossfeeds=[[0,0]], mincov=mincov, ensure_JPGE_size=False)
                assert(len(_p) == 1)
                img = _p[0]
                # print(f"{plots[plt_ix]}: {_end-_start:.4f}", file=sys.stderr)

                ts = transformers[plt_ix](img)
                ts = ts.unsqueeze(0)
                ts = ts.to(device)

                nets[plt_ix].eval()
                _out = nets[plt_ix](ts)

                probs = F.softmax(_out, dim=1)
                conf, pred = torch.max(probs, 1)
                conf = conf.item()
                # print(plots[ix], conf, classes[pred.cpu().numpy()[0]])
                # print(classes[pred.cpu().numpy()[0]] + f":{conf:.4f}", end=",")

                # if imgout != "":
                #     if plots[plt_ix] in _resize:
                #         img = img.resize(_resize[plots[plt_ix]])

                #     img.save(pjoin(imgout, f"{_event_ix}.{plots[plt_ix]}.jpeg"))

                outs.append(_out)

            # ensemble
            allout = sum(outs)
            probs = F.softmax(allout, dim=1)
            conf, pred = torch.max(probs, 1)
            # conf = conf.item()
            # print("sumall", conf, classes[pred.cpu().numpy()[0]], sep=",")
            print(classes[pred.cpu().numpy()[0]], flush=True)


def main(args):
    # print("#", " ".join(sys.argv))
    classes = ["A3", "A5", "CE", "NN", "RI", "SE"]
    if args.cse:
        classes.append("CSE")
        classes.remove("CE")
        classes.remove("SE")
    if args.a:
        classes.append("A")
        classes.remove("A3")
        classes.remove("A5")
    classes = sorted(classes)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"
    print(f"[:deepSpecas:] Using device: {device}", file=sys.stderr)

    plots = []
    models = []
    transformers = []
    if _path := args.pyplot2s:
        _m = deepSpecas._build_model("rs50", 3)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("pyplot2s")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("pyplot2s", 3))

    if _path := args.colmplp2s:
        _m = deepSpecas._build_model("rs50", 4)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("colmplp2s")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("colmplp2s", 4))

    if _path := args.squishstack:
        _m = deepSpecas._build_model("rs50", 3)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("squishstack")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("squishstack", 3))

    if _path := args.squish4d:
        _m = deepSpecas._build_model("rs50", 4)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("squish4d")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("squish4d", 4))

    if _path := args.comb:
        _m = deepSpecas._build_model("rs50", 3)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("comb")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("comb", 3))

    if _path := args.comb4d:
        _m = deepSpecas._build_model("rs50", 4)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("comb4d")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("comb4d", 4))


    predict(
        args.csv, models, plots, transformers, args.mincov, classes, imgout=args.imgout
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Predict using ensemble fo nets")

    parser.add_argument(
        "--zoom", type=int, help="zoom"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Plotting version for each mode [default=1]",
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV to predict [region,BAM_path,GTF_path]",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Produce CAM on classification [DEFAULT: False]",
    )
    parser.add_argument(
        "--imgout",
        type=str,
        default="",
        help="Output dir for saving images [DEFAULT: False]",
    )

    parser.add_argument(
        "--mincov",
        type=int,
        default=0,
        help="minimum covarege to be plot [default=0]",
    )

    parser.add_argument(
        "--pyplot2s",
        type=str,
        required=False,
        default=None,
        help="Trained model on pyplot2s",
    )
    parser.add_argument(
        "--colmplp2s",
        type=str,
        required=False,
        default=None,
        help="Trained model on colmplp2s",
    )
    parser.add_argument(
        "--squishstack",
        type=str,
        required=False,
        default=None,
        help="Trained model on squishstack",
    )
    parser.add_argument(
        "--squish4d",
        type=str,
        required=False,
        default=None,
        help="Trained model on squish4d",
    )

    parser.add_argument(
        "--comb",
        type=str,
        required=False,
        default=None,
        help="Trained model on comb",
    )

    parser.add_argument(
        "--comb4d",
        type=str,
        required=False,
        default=None,
        help="Trained model on comb4d",
    )

    parser.add_argument(
        "--a",
        action="store_true",
        default=False,
        help="Collapse A3 and A5 into A. [DEFAULT: False]",
    )

    parser.add_argument(
        "--cse",
        action="store_true",
        default=False,
        help="Collapse CE and SE into CSE. [DEFAULT: False]",
    )

    args = parser.parse_args()

    if args.imgout:
        makedirs(args.imgout, exist_ok=True)

    main(args)
