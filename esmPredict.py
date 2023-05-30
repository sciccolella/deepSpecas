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

# from torch import topk
# import cv2

# from os import makedirs

# from torchvision.io import read_image

import jitimg
import deepSpecas

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

    _resize = {
        "colmplp": (500, 125),
        "colmplp2s": (500, 125),
        "squishstack": (500, 400),
        "squish4d": (400, 200),
    }

    _event_ix = 0
    print("ix,reg,bam1,bam2", ",".join([f"{p}" for p in plots]), "sumall", sep=",")
    for line in open(args.csv):
        _event_ix += 1
        line = line.strip()
        region, bam1, bam2 = line.split(",")
        print(_event_ix, region, bam1, bam2, sep=",", end=",")
        outs = []
        for plt_ix, _ in enumerate(plots):
            e = jitimg.get_event2s(
                region, bam1, bam2, plot=plots[plt_ix], mincov=mincov
            )
            _start = time.time()
            img = e.plot(
                zoom=20,
                type=plots[plt_ix],
                mode="pil",
                prefix="",
                crossfeeds=[(0, 0)],
                mincov=mincov,
            )[0]
            _end = time.time()
            print(f"{plots[plt_ix]}: {_end-_start:.4f}", file=sys.stderr)

            ts = transformers[plt_ix](img)
            ts = ts.unsqueeze(0)
            ts = ts.to(device)

            # net = net.to(device)
            _out = nets[plt_ix](ts)

            probs = F.softmax(_out, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            # print(plots[ix], conf, classes[pred.cpu().numpy()[0]])
            print(classes[pred.cpu().numpy()[0]] + f":{conf:.4f}", end=",")

            if imgout != "":
                if plots[plt_ix] in _resize:
                    img = img.resize(_resize[plots[plt_ix]])

                img.save(pjoin(imgout, f"{_event_ix}.{plots[plt_ix]}.jpeg"))

            outs.append(_out)

        # ensemble
        allout = sum(outs)
        probs = F.softmax(allout, dim=1)
        conf, pred = torch.max(probs, 1)
        conf = conf.item()
        # print("sumall", conf, classes[pred.cpu().numpy()[0]], sep=",")
        print(classes[pred.cpu().numpy()[0]] + f":{conf:.4f}", flush=True)


def main(args):
    print("#", " ".join(sys.argv))
    classes = sorted(args.classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        # models["pyplot2s"] = [_m, deepSpecas._build_transformer("pyplot2s", 3)]
    if _path := args.colmplp2s:
        _m = deepSpecas._build_model("rs50", 4)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("colmplp2s")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("colmplp2s", 4))
        # models["colmplp2s"] = [_m, deepSpecas._build_transformer("colmplp2s", 4)]
    if _path := args.squishstack:
        _m = deepSpecas._build_model("rs50", 3)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("squishstack")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("squishstack", 3))
        # models["squishstack"] = [_m, deepSpecas._build_transformer("squishstack", 3)]
    if _path := args.squish4d:
        _m = deepSpecas._build_model("rs50", 4)
        _m.fc = nn.Linear(2048, len(classes))
        _m.load_state_dict(torch.load(_path, map_location=device))
        plots.append("squish4d")
        models.append(_m)
        transformers.append(deepSpecas._build_transformer("squish4d", 4))
        # models["squish4d"] = [_m, deepSpecas._build_transformer("squish4d", 4)]

    predict(
        args.csv, models, plots, transformers, args.mincov, classes, imgout=args.imgout
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Predict using ensemble fo nets")

    parser.add_argument(
        "--zoom", type=int, nargs="*", default=[-1], help="zoom [default=-1]"
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
        "--classes",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="Trained model on squish4d",
    )

    args = parser.parse_args()

    if args.imgout:
        makedirs(args.imgout, exist_ok=True)

    main(args)
