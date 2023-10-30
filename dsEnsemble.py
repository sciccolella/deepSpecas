from collections import defaultdict
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
from torch.utils.data import Dataset
from PIL import Image

# from torch import topk
# import cv2

# from os import makedirs

# from torchvision.io import read_image

from functools import partial

import deepSpecas
from plot2s import Event2sample, Event2sampleReads, Event2sampleComb, parse_region
import pandas as pd

import time

MAXREADS = 5_000
PLT_CLS = {
    "pyplot2s": Event2sample,
    "colmplp2s": Event2sample,
    "squishstack": partial(Event2sampleReads, max_reads=MAXREADS),
    "squish4d": partial(Event2sampleReads, max_reads=MAXREADS),
    "comb": partial(Event2sampleComb, max_reads=MAXREADS),
    "comb4d": partial(Event2sampleComb, max_reads=MAXREADS),
}

class EventDataset(Dataset):
    def __init__(
        self,
        df,
        plot: str,
        wd: str = "",
        transform=None,
        target_transform=None,
    ):
        self.imgs = df[df["plot"] == plot]
        self.wd = wd
        self.plot = plot
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs.iloc[idx, 0]
        if self.plot in ["pyplot2s", "squishstack", "comb"]:
            image = Image.open(pjoin(self.wd, img_path)).convert("RGB")
        elif self.plot in ["colmplp2s", "squish4d", "comb4d"]:
            image = Image.open(pjoin(self.wd, img_path)).convert("CMYK")
        else:
            raise ValueError
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image


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

    # nets = [m.to(device) for m in models]

    _event_ix = 0
    print("ix,region,pred", sep=",")
    with torch.no_grad():
        for line in open(csvpath):
            _event_ix += 1
            line = line.strip()
            region, bam1path, bam2path = line.split(",")
            print(_event_ix, region, sep=",", end=",")
            outs = []

            for plt_ix, _plt_name in enumerate(plots):
                _cls = Event2sample
                if plots[plt_ix] in ["squishstack", "squish4d"]:
                    _cls = partial(Event2sampleReads, max_reads=MAXREADS)
                elif plots[plt_ix] in ["comb", "comb4d"]:
                    _cls = partial(Event2sampleComb, max_reads=MAXREADS)

                event = _cls(
                    parse_region(region),
                    pysam.AlignmentFile(bam1path),
                    pysam.AlignmentFile(bam2path),
                )
                for _zoom in args.zoom:
                    _p = event.plot(
                        zoom=_zoom,
                        type=_plt_name,
                        mode="pil",
                        prefix="",
                        crossfeeds=[[0, 0]],
                        mincov=mincov,
                        ensure_JPGE_size=False,
                    )
                    assert len(_p) == 1
                    img = _p[0]
                    # print(f"{plots[plt_ix]}: {_end-_start:.4f}", file=sys.stderr)

                    ts = transformers[plt_ix](img)
                    ts = ts.unsqueeze(0)
                    ts = ts.to(device)

                    _net = models[_plt_name][_zoom].to(device)

                    _net.eval()
                    _out = _net(ts)

                    probs = F.softmax(_out, dim=1)
                    conf, pred = torch.max(probs, 1)
                    conf = conf.item()
                    # print(plots[ix], conf, classes[pred.cpu().numpy()[0]])
                    # print(classes[pred.cpu().numpy()[0]] + f":{conf:.4f}", end=",")

                    # if imgout != "":
                    #     # if plots[plt_ix] in _resize:
                    #     #     img = img.resize(_resize[plots[plt_ix]])

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
    models = defaultdict(dict)
    transformers = []
    if _path := args.pyplot2s:
        # models["pyplot2s"] = {}
        plots.append("pyplot2s")
        transformers.append(deepSpecas._build_transformer("pyplot2s", 3))
        for _zoom in args.zoom:
            _m = deepSpecas._build_model("rs50", 3)
            _m.fc = nn.Linear(2048, len(classes))
            _m.load_state_dict(torch.load(_path.replace("{Z}", f"{_zoom}"), map_location=device))
            # models.append(_m)
            models["pyplot2s"][_zoom] = _m

    if _path := args.colmplp2s:
        # models["colmplp2s"] = {}
        plots.append("colmplp2s")
        transformers.append(deepSpecas._build_transformer("colmplp2s", 4))
        for _zoom in args.zoom:
            _m = deepSpecas._build_model("rs50", 4)
            _m.fc = nn.Linear(2048, len(classes))
            _m.load_state_dict(torch.load(_path.replace("{Z}", f"{_zoom}"), map_location=device))
            # models.append(_m)
            models["colmplp2s"][_zoom] = _m

    if _path := args.squishstack:
        plots.append("squishstack")
        transformers.append(deepSpecas._build_transformer("squishstack", 3))
        for _zoom in args.zoom:
            _m = deepSpecas._build_model("rs50", 3)
            _m.fc = nn.Linear(2048, len(classes))
            _m.load_state_dict(torch.load(_path.replace("{Z}", f"{_zoom}"), map_location=device))
            # models.append(_m)
            models["squishstack"][_zoom] = _m

    if _path := args.squish4d:
        plots.append("squish4d")
        transformers.append(deepSpecas._build_transformer("squish4d", 4))
        for _zoom in args.zoom:
            _m = deepSpecas._build_model("rs50", 4)
            _m.fc = nn.Linear(2048, len(classes))
            _m.load_state_dict(torch.load(_path.replace("{Z}", f"{_zoom}"), map_location=device))
            # models.append(_m)
            models["squish4d"][_zoom] = _m

    if _path := args.comb:
        plots.append("comb")
        transformers.append(deepSpecas._build_transformer("comb", 3))
        for _zoom in args.zoom:
            _m = deepSpecas._build_model("rs50", 3)
            _m.fc = nn.Linear(2048, len(classes))
            _m.load_state_dict(torch.load(_path.replace("{Z}", f"{_zoom}"), map_location=device))
            # models.append(_m)
            models["comb"][_zoom] = _m

    if _path := args.comb4d:
        plots.append("comb4d")
        transformers.append(deepSpecas._build_transformer("comb4d", 4))
        for _zoom in args.zoom:
            _m = deepSpecas._build_model("rs50", 4)
            _m.fc = nn.Linear(2048, len(classes))
            _m.load_state_dict(torch.load(_path.replace("{Z}", f"{_zoom}"), map_location=device))
            # models.append(_m)
            models["comb4d"][_zoom] = _m

    predict(
        args.csv,
        models,
        plots,
        transformers,
        args.mincov,
        classes,
        imgout=args.imgout,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict using ensemble fo nets")

    parser.add_argument("--zoom", nargs="+", type=int, help="zoom")
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

    # parser.add_argument(
    #     "-I",
    #     "--fromimages",
    #     action="store_true",
    #     default=False,
    #     help="Predict from images instead of plotting images JIT [DEFAULT: False]",
    # )

    args = parser.parse_args()

    if args.imgout:
        makedirs(args.imgout, exist_ok=True)

    main(args)
