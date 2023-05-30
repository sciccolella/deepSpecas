import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import sys
import torch.nn as nn
from torch.utils.data import random_split as pytorch_randsplit

# from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
# from torchvision.models import inception_v3, Inception_V3_Weights
# import torch.optim as optim
# from torchvision import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional as F
import copy
import time
from rich.progress import track
from torch.utils.data import Dataset
import pandas as pd
from os.path import join as pjoin
from torch import topk
import cv2

from os import makedirs

from torchvision.io import read_image

import jitimg


class MlplDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        plot: str,
        wd: str = "",
        collapse: bool = False,
        transform=None,
        target_transform=None,
        filter_out: str = "_",
        sample_frac: int = 1,
    ):
        if type(annotations_file) == str:
            self.img_labels = pd.read_csv(
                annotations_file, header=None, names=["path", "event"]
            )
            if filter_out != "_":
                self.img_labels = self.img_labels[
                    self.img_labels["path"].str.contains(filter_out) == False
                ]
            if sample_frac < 1:
                self.img_labels = self.img_labels.sample(frac=sample_frac)
        elif type(annotations_file) == pd.DataFrame:
            self.img_labels = annotations_file
        self.wd = wd
        self.plot = plot
        self.transform = transform
        self.target_transform = target_transform
        self._build_classes(collapse)

    def _build_classes(self, collapse):
        classes = pd.unique(self.img_labels.iloc[:, 1])
        if collapse:
            classes = [x if x not in ["CE", "SE"] else "CSE" for x in classes]
            classes = list(set(classes))
        self.classes = sorted(classes)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        if collapse:
            class_to_idx["CE"] = class_to_idx["CSE"]
            class_to_idx["SE"] = class_to_idx["CSE"]
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        if self.plot in ["pyplot", "colmplp", "pyplot2s", "squishstack"]:
            image = Image.open(pjoin(self.wd, img_path)).convert("RGB")
        elif self.plot in ["colmplp2s", "squish4d"]:
            image = Image.open(pjoin(self.wd, img_path)).convert("CMYK")
        else:
            raise ValueError
            # NOTE: to test
            _np = np.load(pjoin(self.wd, img_path))
            image = torch.from_numpy(_np).float()
        label = self.class_to_idx[self.img_labels.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def _build_transformer(plot: str, n_channels: int, resize: tuple = ()):
    tlist = []
    if plot in ["pyplot", "pyplot2s"]:
        tlist.append(transforms.ToTensor())
    if plot in ["colmplp", "colmplp2s"]:
        tlist.append(transforms.Resize((125, 500)))
        tlist.append(transforms.ToTensor())
    if plot in ["squishstack"]:
        tlist.append(transforms.Resize((400, 500)))
        tlist.append(transforms.ToTensor())
    if plot in ["squish4d"]:
        tlist.append(transforms.Resize((200, 400)))
        tlist.append(transforms.ToTensor())
    if len(resize) > 0:
        tlist.append(transforms.Resize(resize))

    tlist.append(
        transforms.Normalize(
            [0.5 for _ in range(n_channels)], [0.5 for _ in range(n_channels)]
        )
    )
    return transforms.Compose(tlist)


def _build_model(model, n_channels):
    if model == "rs50":
        _m = resnet50(weights=ResNet50_Weights.DEFAULT)
        if n_channels == 3:
            return _m

        og_layer = _m.conv1

        with torch.no_grad():
            new_layer = nn.Conv2d(
                in_channels=n_channels,
                out_channels=og_layer.out_channels,
                kernel_size=og_layer.kernel_size,
                stride=og_layer.stride,
                padding=og_layer.padding,
                bias=og_layer.bias,
            )

            copy_weights = 0  # Here will initialize the weights from new channel with the red channel weights

            # Copying the weights from the old to the new layer
            new_layer.weight[:, : og_layer.in_channels, :, :] = og_layer.weight.clone()

            # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            for i in range(n_channels - og_layer.in_channels):
                channel = og_layer.in_channels + i
                new_layer.weight[:, channel : channel + 1, :, :] = og_layer.weight[
                    :, copy_weights : copy_weights + 1, ::
                ].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)
            _m.conv1 = new_layer
            return _m

    # elif model == "efficientnetv2":
    #     return efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    # elif model == "inv3":
    #     return inception_v3(weights=Inception_V3_Weights.DEFAULT)


def _getCAM(
    features_blob: np.ndarray, weigth_softmax: np.ndarray, class_idx: np.int64
) -> np.ndarray:

    size_upsample = (256, 256)

    _, nc, h, w = features_blob.shape
    cam = weigth_softmax[class_idx].dot(features_blob.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)

    return cam_img


def _saveCAM(
    cam: np.ndarray, w: int, h: int, og_img: np.ndarray, out: str, text: str = None
):
    _hm = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
    res = _hm * 0.3 + og_img * 0.5

    if text:
        cv2.putText(res, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite(out, res)


def load_data(
    folder,
    random_split: bool,
    plot: str,
    n_channels: int,
    collapse: bool = True,
    batch_size: int = 4,
    resize: bool = False,
    shuffle: bool = True,
    wd: str = "",
    sample_frac: int = 1,
    filter_out: str = "_",
):

    _resize = ()
    if resize:
        _resize = (50, 50)
    transform = _build_transformer(plot, n_channels, _resize)

    if type(folder) == pd.DataFrame or folder.endswith(".csv"):
        dataset = MlplDataset(
            folder,
            plot=plot,
            collapse=collapse,
            transform=transform,
            wd=wd,
            sample_frac=sample_frac,
            filter_out=filter_out,
        )
    else:
        sys.exit(1)

    print(
        f"[:load_data:] Dataset loaded. Total images: {len(dataset)}. Classes: {len(dataset.classes)} {dataset.classes}",
        file=sys.stderr,
    )

    if random_split:
        train_dataset, val_dataset = pytorch_randsplit(
            dataset, [t := int((len(dataset) * 0.7)), len(dataset) - t]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
        )

        return (train_loader, val_loader), dataset.classes

    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
        )

        return data_loader, dataset.classes


def train_model(
    model, dataloaders, criterion, optimizer, num_epochs=5, is_inception=False
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    since = time.time()

    model = model.to(device)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in track(dataloaders[phase], description=f"{phase}..."):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # print(inputs.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def validate(val_loader, model, classes=None, hm=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    model = model.to(device)

    correct = 0
    total = 0
    y_pred = []
    y_true = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in track(val_loader, description="Validation..."):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            y_pred.extend(predicted.cpu().numpy())  # Save Prediction
            y_true.extend(labels.cpu().numpy())  # Save Truth

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if hm:
        cf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(cf_matrix, annot=True, xticklabels=classes, yticklabels=classes)
        plt.savefig(hm)

    print(f"Accuracy of the network on the test images: {100 * correct // total} %")


def predict(
    csvpath: str,
    model,
    classes,
    plot: str,
    n_channels: int,
    type: str,
    resize=None,
    explain: bool = False,
    imgout:str = "",
    mincov:int = 0
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if explain:
        # for CAM, not 100% understood what it does
        # https://debuggercafe.com/basic-introduction-to-class-activation-maps-in-deep-learning-using-pytorch/
        # https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
        # https://github.com/chaeyoung-lee/pytorch-CAM/blob/master/main.py
        # https://github.com/chaeyoung-lee/pytorch-CAM/blob/master/update.py
        features_blobs = []

        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        model._modules.get("layer4").register_forward_hook(hook_feature)
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    _resize = ()
    if resize:
        _resize = (50, 50)
    transform = _build_transformer(plot, n_channels, _resize)

    _ix = 0
    for line in open(csvpath):
        _ix += 1
        line = line.strip()
        region, bampath, gtfpath = line.split(",")
        print(_ix, region, bampath, gtfpath)
        # TODO: fix this hardcoded

        if plot in ["pyplot", "colmplp"]:

            if type == "single":
                e = jitimg.get_event(region, bampath, gtfpath, mincov=mincov)
                e.strip()
                print(e.covs.min(), e.covs.max(), e.covs.mean(), np.median(e.covs))
                e.build_zoom(20, mincov=e.covs.mean())
                # e.build_zoom(20)
                img = jitimg.get_img(e.zoom[20], plot)

                if explain:
                    og_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    h, w, _ = og_img.shape

                # FIXME: do for all zooms
                #img.save(f"/home/simone/code/specas/jit.test.{_ix}.png")
                ts = transform(img)
                ts = ts.unsqueeze(0)
                ts = ts.to(device)

                outs = model(ts)
                probs = F.softmax(outs, dim=1)
                conf, pred = torch.max(probs, 1)
                conf = conf.item()
                print(conf, classes[pred.cpu().numpy()[0]])

                if explain:
                    cam = _getCAM(
                        features_blobs[0], weight_softmax, pred.cpu().numpy()[0]
                    )
                    _text = f"{classes[pred.cpu().numpy()[0]]} {conf:.3f}"
                    _saveCAM(
                        cam,
                        w,
                        h,
                        og_img,
                        f"/home/simone/code/specas/jit.test.{_ix}.cam.png",
                        text=_text,
                    )

            elif type == "multi":
                me = jitimg.get_multi_events(region, bampath, gtfpath, mincov=mincov)
                _tix = 0
                for t in me:
                    _tix += 1
                    e = me[t]
                    e.build_zoom(20, mincov=e.covs.mean())
                    img = jitimg.get_img(e.zoom[20], plot)

                    if explain:
                        og_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        h, w, _ = og_img.shape

                    #_debugdout = f"/home/simone/code/specas/debug/multi/{_ix}/"
                    makedirs(_debugdout, exist_ok=True)
                    img.save(pjoin(_debugdout, f"jit.test.{_tix}.{t}.png"))

                    ts = transform(img)
                    ts = ts.unsqueeze(0)
                    ts = ts.to(device)

                    outs = model(ts)
                    probs = F.softmax(outs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    conf = conf.item()
                    print("\t", _tix, t, conf, classes[pred.cpu().numpy()[0]])

                    if explain:
                        cam = _getCAM(
                            features_blobs[0], weight_softmax, pred.cpu().numpy()[0]
                        )
                        _text = f"{classes[pred.cpu().numpy()[0]]} {conf:.3f}"
                        _saveCAM(
                            cam,
                            w,
                            h,
                            og_img,
                            f"/home/simone/code/specas/debug/multi/{_ix}/jit.test.{_tix}.{t}.cam.png",
                            text=_text,
                        )
            else:
                raise ValueError
        elif plot in ["pyplot2s", "colmplp2s", "squishstack", "squish4d"]:
            e = jitimg.get_event2s(region, bampath, gtfpath, plot=plot, mincov=mincov)
            x = e.plot(
                zoom=20, type=plot, mode="pil", prefix="", crossfeeds=[(0, 0)], mincov=mincov
            )

            img = x[0]
            if explain:
                og_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                h, w, _ = og_img.shape

            if imgout != "":
                _imgpathout = pjoin(imgout, f"img.{_ix}.jpeg")
            else:
                _imgpathout = f"/home/simone/code/specas/jit.test.{_ix}.jpeg"

            try:
                img.save(_imgpathout)
            except:
                img = img.resize((400,500))
                #img.save(_imgpathout)

            ts = transform(img)
            ts = ts.unsqueeze(0)
            ts = ts.to(device)

            outs = model(ts)
            probs = F.softmax(outs, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            top6_p, top6_l = topk(probs, len(classes))
            top6_p = top6_p.detach().cpu().numpy()[0]
            top6_l = top6_l.detach().cpu().numpy()[0]

            print(
                f"{_ix}:",
                conf,
                classes[pred.cpu().numpy()[0]],
                "---",
                " ".join(
                    [
                        classes[top6_l[i]] + ":" + f"{top6_p[i]:1.3f}"
                        for i in range(len(top6_l))
                    ]
                ),
            )

            if explain:
                cam = _getCAM(features_blobs[0], weight_softmax, pred.cpu().numpy()[0])
                _text = f"{classes[pred.cpu().numpy()[0]]} {conf:.3f}"
                _saveCAM(
                    cam,
                    w,
                    h,
                    og_img,
                    _imgpathout.replace(".jpeg", ".cam.png"),
                    text=_text,
                )

        else:
            raise ValueError


def main(args):
    if args.plot in ["pyplot", "colmplp", "pyplot2s", "squishstack"]:
        n_channels = 3
    elif args.plot in ["colmplp2s", "squish4d"]:
        n_channels = 4
    else:
        raise ValueError

    model = _build_model(args.model, n_channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[:deepSpecas:] Using device: {device}", file=sys.stderr)

    if args.cmd == "train":
        if not args.val:
            (train_loader, val_loader), classes = load_data(
                args.train,
                random_split=True,
                plot=args.plot,
                n_channels=n_channels,
                batch_size=args.batch_size,
                collapse=args.collapse,
                resize=args.resize,
                wd=args.wd,
                sample_frac=args.sample_frac,
                filter_out=args.no_cf,
            )
        else:

            train_loader, _trainclasses = load_data(
                args.train,
                random_split=False,
                plot=args.plot,
                n_channels=n_channels,
                batch_size=args.batch_size,
                collapse=args.collapse,
                resize=args.resize,
                wd=args.wd,
                sample_frac=args.sample_frac,
                filter_out=args.no_cf,
            )

            val_loader, _valclasses = load_data(
                args.val,
                random_split=False,
                plot=args.plot,
                n_channels=n_channels,
                batch_size=args.batch_size,
                collapse=args.collapse,
                resize=args.resize,
                wd=args.wd,
                sample_frac=args.sample_frac,
                filter_out=args.no_cf,
            )

            # it should not happen that the two classes are different,
            # it would mean to have a bad split, but just in case...
            assert _trainclasses == _valclasses, (
                "Training and validation classes are not equal. "
                + "It might be possible to have a bad split. "
                + "If this is intended, comment this assert."
            )
            classes = _trainclasses

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.fc = nn.Linear(2048, len(classes))

        model, _ = train_model(
            model,
            {"train": train_loader, "val": val_loader},
            criterion,
            optimizer,
            num_epochs=args.epochs,
            is_inception=False,
        )
        if args.net_out:
            torch.save(model.state_dict(), args.net_out)

    if args.cmd in ["train", "validate"]:
        validate(val_loader, model, classes, hm=args.hm)

    if args.cmd == "predict":
        # FIXME
        if args.collapse:
            classes = ["A3", "A5", "CSE", "NN", "RI"]
        else:
            classes = ["A3", "A5", "CE", "NN", "RI", "SE"]

        model.fc = nn.Linear(2048, len(classes))
        model.load_state_dict(torch.load(args.net, map_location=device))

        predict(
            csvpath=args.csv,
            model=model,
            classes=classes,
            plot=args.plot,
            n_channels=n_channels,
            type=args.type,
            resize=args.resize,
            explain=args.explain,
            imgout=args.imgout,
            mincov=args.mincov
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="deepSpecas - ")

    subparsers = parser.add_subparsers(dest="cmd")

    sp_train = subparsers.add_parser("train", help="training mode")

    sp_train.add_argument("--train", type=str, required=True, help="CSV file")
    sp_train.add_argument(
        "--val",
        type=str,
        default=None,
        help="CSV file. If not provided `train` will be randomly split",
    )

    sp_train.add_argument("--epochs", type=int, default=5, help="Training epochs")
    sp_train.add_argument("--batch-size", type=int, default=4, help="Batch size")

    sp_train.add_argument(
        "--sample-frac",
        type=float,
        default=1,
        help="Downsample input when training [Default: 1]",
    )
    sp_train.add_argument(
        "--no-cf", type=str, default="_", help="Ignore a crossfeed value [Default: '_']"
    )

    sp_train.add_argument(
        "-o", "--net-out", type=str, help="Output file for trained network"
    )

    sp_validate = subparsers.add_parser("validate", help="validation mode")
    sp_validate.add_argument("--val", type=str, default=True, help="CSV file.")
    sp_validate.add_argument(
        "--random-split",
        default=False,
        action="store_true",
        help="Create random split for validation from `val`",
    )

    sp_predict = subparsers.add_parser("predict", help="prediction mode")
    # sp_predict.add_argument('-f',
    #                         '--files',
    #                         type=str,
    #                         nargs='+',
    #                         required=True,
    #                         help='Image to predict')
    sp_predict.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV to predict [region,BAM_path,GTF_path]",
    )
    sp_predict.add_argument(
        "--type", type=str, choices=["single", "multi"], help="prediction type"
    )
    sp_predict.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Produce CAM on classification [DEFAULT: False]",
    )
    sp_predict.add_argument(
        "--imgout",
        type=str,
        default="",
        help="Output dir for saving images [DEFAULT: False]",
    )

    sp_predict.add_argument(
        "--mincov",
        type=int,
        default=0,
        help="minimum covarege to be plot [default=0]",
    )

    for p in [sp_validate, sp_predict]:
        p.add_argument("-n", "--net", type=str, required=True, help="Trained net file")

    for p in [sp_train, sp_validate, sp_predict]:
        p.add_argument(
            "-m",
            "--model",
            type=str,
            choices=["rs50", "TODO"],
            default="rs50",
            help="Network model",
        )
        p.add_argument(
            "--wd",
            type=str,
            default="",
            required=False,
            help="Working directory for filenames in CSV. [DEFAULT: '']",
        )
        p.add_argument(
            "--collapse",
            action="store_true",
            default=False,
            help="Collapse A3 and A5 into A. [DEFAULT: False]",
        )
        p.add_argument(
            "--resize",
            action="store_true",
            default=False,
            help="Resize images to half size. [DEFAULT: False]",
        )
        p.add_argument(
            "-p",
            "--plot",
            required=True,
            choices=[
                "pyplot",
                "colmplp",
                "pyplot2s",
                "colmplp2s",
                "squishstack",
                "squish4d",
            ],
            help="Plot mode",
        )
        p.add_argument("--plotversion", default=1, type=int, help="Plot version")
    for p in [sp_train, sp_validate]:
        p.add_argument(
            "--hm", type=str, default="hm.png", help="Heatmap out [default=hm.png]"
        )

    args = parser.parse_args()

    main(args)
