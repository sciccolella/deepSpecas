import numpy as np
import pysam
import itertools
import sys
import re
import argparse

# from skimage.transform import resize as skresize
from rich.progress import track

# from tqdm import tqdm
from os.path import join as pjoin
from os import makedirs

# import matplotlib
from matplotlib import pyplot as plt
import PIL
from typing import Union
from DSutils import parse_gtf, get_transcript

from abc import ABC, abstractmethod
from collections.abc import Iterable


class EventAbstract(ABC):
    @abstractmethod
    def build_zoom(self, flanking: int, mincov: int = 0) -> None:
        pass

    @abstractmethod
    def strip(self) -> None:
        pass


class Event(EventAbstract):
    def __init__(
        self,
        region: tuple,
        bam: pysam.AlignmentFile,
        transcript: np.array,
        mincov: int = 0,
    ) -> None:
        self.region = region
        self.zoom = {}
        self.covs, self.skips = get_coords_from_bam(bam, region, mincov)
        self.mincov = mincov
        if type(transcript) == str:
            _exons = parse_gtf(transcript)  # TODO: remove this option
            transcript = get_transcript(_exons, region[1], region[2])
        self.transcript = transcript

    def build_zoom(self, flanking: int, mincov: int = 0) -> None:
        if flanking == -1:
            self.zoom[-1] = (self.covs, self.skips, self.transcript)
            return
        _cov_count = np.where(self.covs <= mincov, 0, 1)
        _trans_count = np.where(self.transcript == 0, 0, 1)

        tot = _cov_count + _trans_count
        blocks = list(to_ranges((tot == 0).nonzero()[0]))
        if len(blocks) > 0:
            _bs = list()
            for b in blocks:
                # print(b)
                # if b[0] + flanking > b[1] - flanking:
                #     s = b[0]
                #     e = b[1]
                # else:
                if not b[0] + flanking > b[1] - flanking:
                    s = b[0] + flanking
                    e = b[1] - flanking

                    _bs.append(np.r_[s:e])
            if len(_bs) > 0:
                rm = np.concatenate(_bs)

                self.zoom[flanking] = (
                    np.delete(self.covs, rm, axis=0),
                    np.delete(self.skips, rm, axis=0),
                    np.delete(self.transcript, rm, axis=0),
                )
                return

        self.zoom[flanking] = (self.covs, self.skips, self.transcript)

    def lstrip(self) -> None:
        _cov_count = np.where(self.transcript == 0, 0, 1)
        blocks = list(to_ranges((_cov_count == 0).nonzero()[0]))
        if len(blocks) > 0:
            bf = blocks[0]
            if bf[0] == 0 and bf[1] != 0:
                rm = np.r_[0 : bf[1] + 1]
                self.covs = np.delete(self.covs, rm, axis=0)
                self.skips = np.delete(self.skips, rm, axis=0)
                self.transcript = np.delete(self.transcript, rm, axis=0)

    def rstrip(self) -> None:
        _cov_count = np.where(self.transcript == 0, 0, 1)
        blocks = list(to_ranges((_cov_count == 0).nonzero()[0]))
        if len(blocks) > 0:
            bl = blocks[-1]
            if bl[1] == self.covs.shape[0] - 1:
                rm = np.r_[bl[0] : self.covs.shape[0] - 1]
                self.covs = np.delete(self.covs, rm, axis=0)
                self.skips = np.delete(self.skips, rm, axis=0)
                self.transcript = np.delete(self.transcript, rm, axis=0)

    def strip(self) -> None:
        self.lstrip()
        self.rstrip()


class Event2sample(EventAbstract):
    def __init__(
        self,
        region: tuple,
        bam1: pysam.AlignmentFile,
        bam2: pysam.AlignmentFile,
        mincov: int = 0,
    ) -> None:
        self.region = region
        self.zoom = {}
        self.covs_1, self.skips_1 = get_coords_from_bam(bam1, region, mincov)
        self.covs_2, self.skips_2 = get_coords_from_bam(bam2, region, mincov)

        self.mincov = mincov

    def build_zoom(self, flanking: int, mincov: int = 0) -> None:
        if flanking == -1:
            self.zoom[-1] = (self.covs_1, self.skips_1, self.covs_2, self.skips_2)
            return

        _cov1_counts = np.where(self.covs_1 <= mincov, 0, 1)
        _cov2_counts = np.where(self.covs_2 <= mincov, 0, 1)

        tot = _cov1_counts + _cov2_counts

        blocks = list(to_ranges((tot == 0).nonzero()[0]))
        # print(blocks)
        if len(blocks) > 0:
            _bs = list()
            for b in blocks:
                if not b[0] + flanking > b[1] - flanking:
                    s = b[0] + flanking
                    e = b[1] - flanking

                    _bs.append(np.r_[s:e])
            if len(_bs) > 0:
                rm = np.concatenate(_bs)

                self.zoom[flanking] = (
                    np.delete(self.covs_1, rm, axis=0),
                    np.delete(self.skips_1, rm, axis=0),
                    np.delete(self.covs_2, rm, axis=0),
                    np.delete(self.skips_2, rm, axis=0),
                )
                return

        self.zoom[flanking] = (self.covs_1, self.skips_1, self.covs_2, self.skips_2)

    def strip(self) -> None:
        pass

    def plot(
        self,
        zoom: int,
        type: str,
        mode: str,
        prefix: str,
        crossfeeds: list[tuple[float]],
        **kwargs,
    ) -> Union[None, list[PIL.Image.Image]]:

        if not zoom in self.zoom:
            self.build_zoom(zoom, **kwargs)

        c1, s1, c2, s2 = self.zoom[zoom]
        __ret = list()

        if type == "pyplot2s":
            __plot = Event2sample._pyplot
            ext = "png"
        elif type == "colmplp2s":
            __plot = Event2sample._colmplp
            ext = "jpeg"
        else:
            raise ValueError

        for cf1, cf2 in crossfeeds:
            _c1, _s1, _c2, _s2 = Event2sample._crossfeed_pileups(
                c1, s1, c2, s2, cf1, cf2
            )

            fout = f"{prefix}.cf_{cf1}-{cf2}.{ext}"
            __ret.append(__plot(_c1, _s1, _c2, _s2, out=fout, mode=mode))

        return __ret

    @staticmethod
    def _pyplot(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
    ) -> Union[str, PIL.Image.Image]:

        x = list(range(len(s1)))
        plt.subplot(2, 1, 1)
        plt.bar(x, s1, color="blue", width=1.0)
        plt.bar(x, c1, bottom=s1, color="red", width=1.0)
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.bar(x, s2, color="blue", width=1.0)
        plt.bar(x, c2, bottom=s2, color="red", width=1.0)
        plt.axis("off")

        plt.tight_layout()

        if mode == "savefig":
            plt.savefig(out)
            plt.close()
            return out
        elif mode == "pil":
            fig = plt.gcf()
            fig.canvas.draw()
            ret = PIL.Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )

            plt.close()
            return ret
        else:
            raise ValueError

    @staticmethod
    def _colmplp(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
    ) -> Union[str, PIL.Image.Image]:

        x = list(range(len(c1)))

        h = max(c1.max(), c2.max(), s1.max(), s2.max())
        c1mat = _getmat(c1, h)
        s1mat = _getmat(s1, h)
        c2mat = _getmat(c2, h)
        s2mat = _getmat(s2, h)

        cmyk = np.dstack((c1mat, s1mat, c2mat, s2mat))
        img = PIL.Image.fromarray(cmyk, "CMYK")

        if mode == "savefig":
            img.save(out)
            return out
        elif mode == "pil":
            return img
        else:
            raise ValueError

    @staticmethod
    def _crossfeed_pileups(
        c1: np.array,
        s1: np.array,
        c2: np.array,
        s2: np.array,
        perc_1: float,
        perc_2: float,
    ) -> tuple[np.array]:

        if perc_2 == 0:
            _c1 = c1
            _s1 = s1
        else:
            _c1 = c1.copy()
            _s1 = s1.copy()
            _c1 += (c2 * perc_2).astype(np.uint32)
            _s1 += (s2 * perc_2).astype(np.uint32)

        if perc_1 == 0:
            _c2 = c2
            _s2 = s2
        else:
            _c2 = c2.copy()
            _s2 = s2.copy()
            _c2 += (c1 * perc_1).astype(np.uint32)
            _s2 += (s1 * perc_1).astype(np.uint32)

        return (_c1, _s1, _c2, _s2)


class Event2sampleReads(EventAbstract):
    def __init__(
        self,
        region: tuple,
        bam1: pysam.AlignmentFile,
        bam2: pysam.AlignmentFile,
        max_reads: int = -1,
    ) -> None:
        self.region = region
        self.zoom = {}
        self.covs_1, self.skips_1 = get_reads_from_bam(bam1, region, max_reads)
        self.covs_2, self.skips_2 = get_reads_from_bam(bam2, region, max_reads)

        self.max_reads = max_reads

    def strip(self):
        pass

    def plot(
        self,
        zoom: int,
        type: str,
        mode: str,
        prefix: str,
        crossfeeds: list[tuple[float]],
        **kwargs,
    ) -> Union[None, list[PIL.Image.Image]]:

        if not zoom in self.zoom:
            self.build_zoom(zoom, **kwargs)

        __ret = list()

        if type == "squishstack":
            __plot = Event2sampleReads._squished_stacked
            ext = "png"
        elif type == "squish4d":
            __plot = Event2sampleReads._squished_4d
            ext = "jpeg"
        else:
            raise ValueError

        for cf1, cf2 in crossfeeds:
            _c1, _s1, _c2, _s2 = Event2sampleReads._crossfeed(
                *self.zoom[zoom], cf1, cf2
            )
            if cf2 > 0:
                _c1, _s1 = Event2sampleReads._row_reoder_bystart(_c1, _s1)
            if cf1 > 0:
                _c2, _s2 = Event2sampleReads._row_reoder_bystart(_c2, _s2)

            fout = f"{prefix}.cf_{cf1}-{cf2}.{ext}"
            __ret.append(__plot(_c1, _s1, _c2, _s2, out=fout, mode=mode))

        return __ret

    def build_zoom(self, flanking: int, mincov: int = 0) -> None:
        if flanking == -1:
            self.zoom[-1] = (self.covs_1, self.skips_1, self.covs_2, self.skips_2)
            return
        
        _cov1_counts = np.where(self.covs_1 == 0, 0, 1)
        _cov1_counts = np.dot(np.ones(_cov1_counts.shape[0]), _cov1_counts)
        _cov2_counts = np.where(self.covs_2 == 0,  0, 1)
        _cov2_counts = np.dot(np.ones(_cov2_counts.shape[0]), _cov2_counts)

        # print(list(_cov1_counts))

        tot = _cov1_counts + _cov2_counts
        tot = np.where(tot < mincov, 0, tot)

        blocks = list(to_ranges((tot == 0).nonzero()[0]))
        # print(blocks)
        if len(blocks) > 0:
            _bs = list()
            for b in blocks:
                if not b[0] + flanking > b[1] - flanking:
                    s = b[0] + flanking
                    e = b[1] - flanking

                    _bs.append(np.r_[s:e])
            if len(_bs) > 0:
                rm = np.concatenate(_bs)

                self.zoom[flanking] = (
                    np.delete(self.covs_1, rm, axis=1),
                    np.delete(self.skips_1, rm, axis=1),
                    np.delete(self.covs_2, rm, axis=1),
                    np.delete(self.skips_2, rm, axis=1),
                )
                return

        self.zoom[flanking] = (self.covs_1, self.skips_1, self.covs_2, self.skips_2)

    @staticmethod
    def _squished_stacked(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
    ) -> Union[str, PIL.Image.Image]:

        stack = PIL.Image.new("RGB", (c1.shape[1], c1.shape[0] + c2.shape[0]))

        rgb1 = np.dstack((c1, s1, np.zeros_like(c1)))
        img1 = PIL.Image.fromarray(rgb1, "RGB")

        rgb2 = np.dstack((c2, s2, np.zeros_like(c2)))
        img2 = PIL.Image.fromarray(rgb2, "RGB")

        stack.paste(img1, (0, 0))
        stack.paste(img2, (0, c1.shape[0]))

        if mode == "savefig":
            stack.save(out)
            return out
        elif mode == "pil":
            return stack
        else:
            raise ValueError

    @staticmethod
    def _squished_4d(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
    ) -> Union[str, PIL.Image.Image]:

        _c1, _s1, _c2, _s2 = Event2sampleReads.pad(c1, s1, c2, s2)
        cmyk = np.dstack((_c1, _s1, _c2, _s2))
        img = PIL.Image.fromarray(cmyk, "CMYK")

        if mode == "savefig":
            img.save(out)
            return out
        elif mode == "pil":
            return img
        else:
            raise ValueError

    @staticmethod
    def pad(c1: np.ndarray, s1: np.ndarray, c2: np.ndarray, s2: np.ndarray) -> None:
        if c1.shape[0] < c2.shape[0]:
            padsize = ((0, c2.shape[0] - c1.shape[0]), (0, 0))
            return (
                np.pad(c1, padsize, "constant"),
                np.pad(s1, padsize, "constant"),
                c2,
                s2,
            )
        else:
            padsize = ((0, c1.shape[0] - c2.shape[0]), (0, 0))
            return (
                c1,
                s1,
                np.pad(c2, padsize, "constant"),
                np.pad(s2, padsize, "constant"),
            )

    @staticmethod
    def _row_reoder_bystart(c: np.ndarray, s: np.ndarray):
        starts = []
        startmax = c.shape[1]
        for i in range(c.shape[0]):
            _csnz = np.nonzero(c[i, :])[0]
            _ssnz = np.nonzero(s[i, :])[0]
            _cs = _csnz[0] if len(_csnz) > 0 else startmax
            _ss = _ssnz[0] if len(_ssnz) > 0 else startmax
            starts.append(min(_cs, _ss))

        index = np.array(sorted(range(len(starts)), key=lambda i: starts[i]))

        return c[index], s[index]

    # @staticmethod
    # def _row_reorder(c: np.ndarray, s: np.ndarray):
    #     start_end = []
    #     startmax = c.shape[1]
    #     for i in range(c.shape[0]):
    #         _csnz = np.nonzero(c[i, :])[0]
    #         _ssnz = np.nonzero(s[i, :])[0]
    #         _cs = _csnz[0] if len(_csnz) > 0 else startmax
    #         _ss = _ssnz[0] if len(_ssnz) > 0 else startmax
    #         _start = min(_cs, _ss)

    #         _ce = _csnz[-1] if len(_csnz) > 0 else 0
    #         _se = _ssnz[-1] if len(_ssnz) > 0 else 0
    #         _end = max(_ce, _se)
    #         start_end.append((_start, _end))

    #     # print(sorted(start_end, key=lambda x: (x[0], -x[1])))

    #     return np.array(
    #         sorted(
    #             range(len(start_end)), key=lambda i: (start_end[i][0],0)
    #         )
    #     )

    @staticmethod
    def _crossfeed(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        perc_1: float,
        perc_2: float,
    ) -> tuple[np.ndarray]:
        if perc_2 == 0:
            _c1 = c1
            _s1 = s1
        else:
            _c1 = c1.copy()
            _s1 = s1.copy()
            _ac = c2[
                np.random.choice(c2.shape[0], int(c2.shape[0] * perc_2), replace=False),
                :,
            ]
            _as = s2[
                np.random.choice(s2.shape[0], int(s2.shape[0] * perc_2), replace=False),
                :,
            ]
            _c1 = np.vstack((_c1, _ac))
            _s1 = np.vstack((_s1, _as))

        if perc_1 == 0:
            _c2 = c2
            _s2 = s2
        else:
            _c2 = c2.copy()
            _s2 = s2.copy()
            _ac = c1[
                np.random.choice(c1.shape[0], int(c1.shape[0] * perc_2), replace=False),
                :,
            ]
            _as = s1[
                np.random.choice(s1.shape[0], int(s1.shape[0] * perc_2), replace=False),
                :,
            ]
            _c2 = np.vstack((_c2, _ac))
            _s2 = np.vstack((_s2, _as))

        return (_c1, _s1, _c2, _s2)


# def __pyplot_debug(data, out):

#     for _ix, d in enumerate(data):
#         plt.subplot(len(data), 1, _ix + 1)
#         x = range(len(d[0]))
#         plt.bar(x, d[1], color="blue", width=1.0)
#         plt.bar(x, d[0], bottom=d[1], color="red", width=1.0)
#     plt.savefig(out)
#     plt.close()


def to_ranges(iterable: Iterable) -> Iterable:
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


# import uuid
# import random

# random.seed(21)

# def crossfeed_bams(bam1: pysam.AlignmentFile, bam2: pysam.AlignmentFile,
#                    perc: float) -> str:
#     out = f"/tmp/{uuid.uuid4().hex}.bam"
#     obam = pysam.AlignmentFile(out, "wb", template=bam1)
#     for read in bam1.fetch():
#         obam.write(read)
#     for read in bam2.fetch():
#         if random.random() < perc:
#             read.query_name = read.query_name + "_crossfeed"
#             obam.write(read)
#     obam.close()
#     print(out)
#     pysam.sort("-o", f"{out}.sorted.bam", out)
#     pysam.index(f"{out}.sorted.bam")
#     return f"{out}.sorted.bam"


def get_coords_from_bam(bam: pysam.AlignmentFile, region: tuple, mincov: int = 0):
    # xs = np.arange(*region[1:], dtype=np.uint8)
    covs = np.zeros(region[2] - region[1], dtype=np.uint32)
    skips = np.zeros_like(covs, dtype=np.uint32)

    for pileupcolumn in bam.pileup(
        *region,
        stepper="nofilter",
        flag_filter=0,
        ignore_orphans=False,
        min_base_quality=0,
    ):
        if (
            (pos := pileupcolumn.reference_pos) > region[1]
            and pos < region[2]
            and (cov := len(pileupcolumn.pileups)) > mincov
        ):
            skipcount = 0
            for pileupread in pileupcolumn.pileups:
                skipcount += pileupread.is_refskip
            # print(pos, pileupcolumn.nsegments, len(pileupcolumn.pileups), skipcount)
            # if skipcount != cov:
            covs[pos - region[1]] = cov - skipcount
            skips[pos - region[1]] = skipcount

    return covs, skips


def get_reads_from_bam(bam: pysam.AlignmentFile, region: tuple, max_reads: int = -1):
    rstart, rend = region[1:]
    regionlen = rend - rstart
    subsample = -1

    _rows = bam.count(*region)
    if max_reads > 0 and _rows > max_reads:
        subsample = (max_reads * 0.9) / _rows
        print(_rows, subsample)
        _rows = max_reads
    _mat_shape = (_rows, regionlen)
    # index = dict()
    mat_cov = np.zeros(_mat_shape, dtype=np.uint8)
    mat_skips = np.zeros(_mat_shape, dtype=np.uint8)

    # print(mat.shape)
    ix = -1
    for read in bam.fetch(*region):
        if subsample > 0:
            rand_u = np.random.rand()
            if rand_u > subsample:
                continue
        ix += 1

        ix_cigar = 0
        ix_alnd_block = 0
        blocks = read.get_blocks()
        fblocks = []
        for b in blocks:
            bstart, bend = b
            bstart = max(0, bstart - rstart - 1)  # CHECK if -1
            bstart = min(bstart, regionlen)
            bend = max(0, bend - rstart - 1)  # CHECK if -1
            bend = min(bend, regionlen)
            fblocks.append((bstart, bend))
        # print(fblocks, read.cigartuples)
        while ix_cigar < len(read.cigartuples):
            if read.cigartuples[ix_cigar][0] == 0:
                # print("[block 0]", ix_cigar, ix_alnd_block)
                # This an aligned block
                block = fblocks[ix_alnd_block]
                bstart, bend = block
                ix_alnd_block += 1
                # if bend == 0:
                #     pass
                # else:
                # mat[readix][np.r_[bstart:bend]] = 255
                mat_cov[ix][np.r_[bstart:bend]] = 255

                # TODO: check if out of region on right (maybe?)
                # ix_cigar += 1
            if read.cigartuples[ix_cigar][0] == 3:
                # print("[block 3]", ix_cigar, ix_alnd_block)
                # skip after aligned block
                skipreg = (fblocks[ix_alnd_block - 1][1], fblocks[ix_alnd_block][0])
                if skipreg[1] > 0:
                    mat_skips[ix][np.r_[skipreg[0] : skipreg[1]]] = 255
            ix_cigar += 1

    return mat_cov, mat_skips


def parse_region(string: str) -> tuple:
    reg = re.match(r"(?P<chr>[\w\d]+):(?P<start>\d+)-(?P<end>\d+)", string)
    if not reg:
        print(
            "Unable to read region {string}. Please check the format. Ignoring it",
            file=sys.stderr,
        )
        sys.exit(1)
    return [reg.group("chr"), int(reg.group("start")), int(reg.group("end"))]


def pyplot_plt(
    event_data: tuple[np.array], out: str, mode: str
) -> Union[None, list[PIL.Image.Image]]:
    _color = ["red", "blue", "green"]
    ret = None

    for _ix, data in enumerate(event_data):
        plt.subplot(3, 1, _ix + 1)
        plt.plot(data, color=_color[_ix])
        plt.fill_between(range(len(data)), data, color=_color[_ix])
        plt.axis("off")
    plt.tight_layout()
    if mode == "savefig":
        plt.savefig(out)
    elif mode == "pil":
        fig = plt.gcf()
        fig.canvas.draw()
        ret = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
    plt.close()

    if mode == "pil":
        return ret


def _getmat(data: np.array, h: int, fill: int = 255) -> np.array:
    w = data.shape[0]
    # if np.max(data) == 0:
    #     return np.zeros(size).astype(int)

    a = np.zeros((h, w), dtype=np.uint8)
    a = a.T
    for i in range(a.shape[0]):
        # top = min(int(data[i]), h)
        top = int(data[i])
        ii = np.r_[0:top]
        a[i][[ii]] = fill
    a = a.T
    a = np.flip(a, axis=0)

    return a


# def _getmat_externalmax(data: np.array,
#                         emax: np.array,
#                         h: int,
#                         fill: int = 255) -> np.array:
#     w = data.shape[0]
#     # if np.max(data) == 0:
#     #     return np.zeros(size).astype(int)

#     a = np.zeros((h, w), dtype=np.uint8)
#     a = a.T
#     for i in range(a.shape[0]):
#         top = min(int(emax[i]), h) if data[i] > 0 else 0
#         ii = np.r_[0:top]
#         a[i][[ii]] = fill
#     a = a.T
#     a = np.flip(a, axis=0)

#     return a


def _getmat_fixedmax(data: np.array, fmax: int, h: int, fill: int = 255) -> np.array:
    w = data.shape[0]
    # if np.max(data) == 0:
    #     return np.zeros(size).astype(int)

    a = np.zeros((h, w), dtype=np.uint8)
    a = a.T
    for i in range(a.shape[0]):
        top = min(fmax, h) if data[i] > 0 else 0
        ii = np.r_[0:top]
        a[i][[ii]] = fill
    a = a.T
    a = np.flip(a, axis=0)

    return a


def colmplp_plt(
    event_data: tuple[np.array], out: str, mode: str
) -> Union[None, list[PIL.Image.Image]]:

    # RED     (255,0,0)       -> covs only
    # GREEN   (0,255,0)       -> skip only
    # BLUE    (0,0,255)       -> gtf only
    # PURPLE  (255,0,255)     -> covs + gtf
    # CYAN    (0,255,255)     -> skips + gtf

    h = max(event_data[0].max(), event_data[1].max())
    covmat = _getmat(event_data[0], h)
    skipmat = _getmat(event_data[1], h)
    tmat = _getmat_fixedmax(event_data[2], h, event_data[0].max())

    rgb = np.dstack((covmat, skipmat, tmat))
    img = PIL.Image.fromarray(rgb, "RGB")

    if mode == "savefig":
        img.save(out)
    elif mode == "pil":
        return img


def _plt_np(arrs, out):
    for i, a in enumerate(arrs):
        plt.subplot(len(arrs), 1, i + 1)
        plt.plot(a)
    plt.savefig(out)
    plt.close()


def main(args):
    _modemap = {"pyplot": pyplot_plt, "colmplp": colmplp_plt}
    _modemapext = {"pyplot": "png", "colmplp": "png"}
    # _eventCLS = {"1s1t": Event, "2s": Event2sample}
    __ix = 0
    __max_ix = 20
    _csvin = None

    _logfile = None
    if args.log:
        _logfile = open(args.log, "w")

    outcsv = open(args.outcsv, "w")

    with open(args.input, "r") as fin:
        _csvin = fin.readlines()

    # for line in track(_csvin[:__max_ix], total=__max_ix):
    #     __ix += 1
    for line in track(_csvin):
        __ix += 1
        if _logfile:
            _logfile.write(line)
        line = line.strip()

        if args.mode in ["pyplot", "colmplp"]:
            _region, _bam, _gtf, _event = line.split(",")

            region = parse_region(_region)
            bam = pysam.AlignmentFile(pjoin(args.datadir, _bam), "rb")
            _exons = parse_gtf(pjoin(args.datadir, _gtf))
            transcript = get_transcript(_exons, region[1], region[2])
            event = Event(region, bam, transcript, mincov=args.mincov)

            bam.close()
            event.strip()

            # TODO: fix this to be uniform with the other

            fname = f"{_region.replace(':','_')}.{_bam.split('/')[-1].split('.')[1]}.{_gtf.split('/')[-1].split('.')[0]}"
            ext = _modemapext[args.mode]
            for _zoom in args.zoom:
                event.build_zoom(_zoom)
                _modemap[args.mode](
                    event.zoom[_zoom],
                    out=pjoin(args.outdir, f"zoom{_zoom}", f"{fname}.{ext}"),
                    mode="savefig",
                )
            _csv_line = f"{fname}.{ext},{_event}\n"
            outcsv.write(_csv_line)
            outcsv.flush()

        elif args.mode in ["pyplot2s", "colmplp2s"]:
            _region, _bam1, _bam2, _event = line.split(",")
            # print(_bam1, _bam2, _event)

            region = parse_region(_region)
            bam1 = pysam.AlignmentFile(pjoin(args.datadir, _bam1), "rb")
            bam2 = pysam.AlignmentFile(pjoin(args.datadir, _bam2), "rb")
            event = Event2sample(region, bam1, bam2, mincov=args.mincov)

            # NOTE: no strip is done on these events. Should we? How?

            bam1.close()
            bam2.close()

            prefix = f"{_region.replace(':','_')}.{_bam1.split('/')[-1].split('.')[1]}.{_bam2.split('/')[-1].split('.')[1]}"

            for _zoom in args.zoom:
                out_files = event.plot(
                    _zoom,
                    type=args.mode,
                    mode="savefig",
                    prefix=pjoin(args.outdir, f"zoom{_zoom}", prefix),
                    crossfeeds=[(0, 0), (0, 0.2), (0.2, 0), (0.2, 0.2)],
                )
                for of in out_files:
                    outcsv.write(f"{of.lstrip(args.outdir)},{_event}\n")
                outcsv.flush()
        elif args.mode in ["squishstack", "squish4d"]:
            _region, _bam1, _bam2, _event = line.split(",")
            # print(_bam1, _bam2, _event)

            region = parse_region(_region)
            bam1 = pysam.AlignmentFile(pjoin(args.datadir, _bam1), "rb")
            bam2 = pysam.AlignmentFile(pjoin(args.datadir, _bam2), "rb")
            event = Event2sampleReads(
                region, bam1, bam2, max_reads=-1  # TODO: fix this
            )

            # NOTE: no strip is done on these events. Should we? How?

            bam1.close()
            bam2.close()
            
            prefix = f"{_region.replace(':','_')}.{_bam1.split('/')[-1].split('.')[1]}.{_bam2.split('/')[-1].split('.')[1]}"

            for _zoom in args.zoom:
                out_files = event.plot(
                    _zoom,
                    type=args.mode,
                    mode="savefig",
                    prefix=pjoin(args.outdir, f"zoom{_zoom}", prefix),
                    crossfeeds=[(0, 0), (0, 0.2), (0.2, 0), (0.2, 0.2)],
                )
                for of in out_files:
                    outcsv.write(f"{of.lstrip(args.outdir)},{_event}\n")
                outcsv.flush()
        else:
            raise ValueError

    if _logfile:
        _logfile.close()

    outcsv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot figures from mpileup")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="CSV file of events"
    )
    parser.add_argument("-o", "--outdir", type=str, help="output directory")
    parser.add_argument(
        "-d",
        "--datadir",
        type=str,
        default="",
        help="root directory containing bams and gtfs",
    )
    parser.add_argument(
        "--outcsv", type=str, required=True, help="Output CSV file of events"
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="pyplot",
        choices=["pyplot", "colmplp", "pyplot2s", "colmplp2s", "squishstack", "squish4d"],
        help="Plot type",
    )

    parser.add_argument(
        "--mincov",
        type=float,
        default=0.0,
        help="minimum covarege to be plot [default=0]",
    )
    parser.add_argument(
        "--zoom", type=int, nargs="*", default=[-1], help="zoom [default=-1]"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Plotting version for each mode [default=1]",
    )
    parser.add_argument("--log", type=str, default=None, help="Log output file")
    args = parser.parse_args()

    for z in args.zoom:
        makedirs(pjoin(args.outdir, f"zoom{z}"), exist_ok=True)

    main(args)
