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
from os.path import basename

# import matplotlib
from matplotlib import pyplot as plt
import PIL
from typing import Union

from abc import ABC, abstractmethod
from collections.abc import Iterable

RESIZE_SIZE = {
    "pyplot2s": (400, 1000),
    "colmplp2s": (200, 1000),
    "squishstack": (400, 1000),
    "squish4d": (200, 1000),
    "comb": (800, 1000),
    "comb4d": (400, 1000),
}


class EventAbstract(ABC):
    @abstractmethod
    def build_zoom(self, flanking: int, mincov: int = 0) -> None:
        pass

    @abstractmethod
    def strip(self) -> None:
        pass


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

    def build_zoom(self, flanking: int, mincov: int = 0, **_) -> None:
        if flanking == -1:
            self.zoom[-1] = (self.covs_1, self.skips_1, self.covs_2, self.skips_2)
            return

        _cov1_counts = np.where(self.covs_1 <= mincov, 0, 1)
        _cov2_counts = np.where(self.covs_2 <= mincov, 0, 1)

        tot = _cov1_counts + _cov2_counts

        blocks = list(to_ranges((tot == 0).nonzero()[0]))
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
        _cov_counts = self.covs_1 + self.covs_2
        blocks = list(to_ranges((_cov_counts == 0).nonzero()[0]))
        if len(blocks) > 0:
            torm = []
            to_check = [blocks[0]] if len(blocks) == 1 else [blocks[0], blocks[-1]]
            for bl in to_check:
                if bl[0] == 0:
                    torm.append(np.r_[0 : bl[1] + 1])
                elif bl[1] == self.covs_1.shape[0] - 1:
                    torm.append(np.r_[bl[0] : bl[1]])
                else:
                    continue
            if len(torm) > 0:
                rm = np.concatenate(torm)
                self.covs_1 = np.delete(self.covs_1, rm, axis=0)
                self.skips_1 = np.delete(self.skips_1, rm, axis=0)
                self.covs_2 = np.delete(self.covs_2, rm, axis=0)
                self.skips_2 = np.delete(self.skips_2, rm, axis=0)

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
            __ret.append(__plot(_c1, _s1, _c2, _s2, out=fout, mode=mode, **kwargs))

        return __ret

    @staticmethod
    def _pyplot(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
        resize: bool = False,
        **_,
    ) -> Union[str, PIL.Image.Image]:
        h = max(c1.max(), c2.max(), s1.max(), s2.max())
        c1mat = _getmat(c1, h)
        s1mat = _getmat(s1, h)
        c2mat = _getmat(c2, h)
        s2mat = _getmat(s2, h)

        stack = PIL.Image.new("RGB", (c1mat.shape[1], h * 2))

        rgb1 = np.dstack((c1mat, s1mat, np.zeros_like(c1mat)))
        img1 = PIL.Image.fromarray(rgb1, "RGB")

        rgb2 = np.dstack((c2mat, s2mat, np.zeros_like(c2mat)))
        img2 = PIL.Image.fromarray(rgb2, "RGB")

        stack.paste(img1, (0, 0))
        stack.paste(img2, (0, h))

        if resize:
            stack = stack.resize(RESIZE_SIZE["pyplot2s"][::-1])

        if mode == "savefig":
            stack.save(out)
            return out
        elif mode == "pil":
            return stack
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
        resize: bool = False,
        ensure_JPGE_size: bool = True,
        **_,
    ) -> Union[str, PIL.Image.Image]:
        h = max(c1.max(), c2.max(), s1.max(), s2.max())
        c1mat = _getmat(c1, h)
        s1mat = _getmat(s1, h)
        c2mat = _getmat(c2, h)
        s2mat = _getmat(s2, h)

        cmyk = np.dstack((c1mat, s1mat, c2mat, s2mat))
        img = PIL.Image.fromarray(cmyk, "CMYK")

        if resize:
            img = img.resize(RESIZE_SIZE["colmplp2s"][::-1])

        if ensure_JPGE_size:
            while img.width * img.height > 65_500:  # max size for JPGE
                img = img.resize((img.width // 2, img.height // 2))

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
        _cov1_counts = np.where(self.covs_1 == 0, 0, 1)
        _cov1_counts = np.dot(np.ones(_cov1_counts.shape[0]), _cov1_counts)
        _cov2_counts = np.where(self.covs_2 == 0, 0, 1)
        _cov2_counts = np.dot(np.ones(_cov2_counts.shape[0]), _cov2_counts)

        tot = _cov1_counts + _cov2_counts
        tot = np.where(tot == 0, 0, tot)

        blocks = list(to_ranges((tot == 0).nonzero()[0]))

        if len(blocks) > 0:
            torm = []
            to_check = [blocks[0]] if len(blocks) == 1 else [blocks[0], blocks[-1]]
            for bl in to_check:
                if bl[0] == 0:
                    torm.append(np.r_[0 : bl[1] + 1])
                elif bl[1] == self.covs_1.shape[1] - 1:
                    torm.append(np.r_[bl[0] : bl[1]])
                else:
                    continue
            if len(torm) > 0:
                rm = np.concatenate(torm)
                self.covs_1 = np.delete(self.covs_1, rm, axis=1)
                self.skips_1 = np.delete(self.skips_1, rm, axis=1)
                self.covs_2 = np.delete(self.covs_2, rm, axis=1)
                self.skips_2 = np.delete(self.skips_2, rm, axis=1)

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
            __ret.append(__plot(_c1, _s1, _c2, _s2, out=fout, mode=mode, **kwargs))

        return __ret

    def build_zoom(self, flanking: int, mincov: int = 0, **_) -> None:
        if flanking == -1:
            self.zoom[-1] = (self.covs_1, self.skips_1, self.covs_2, self.skips_2)
            return

        _cov1_counts = np.where(self.covs_1 == 0, 0, 1)
        _cov1_counts = np.dot(np.ones(_cov1_counts.shape[0]), _cov1_counts)
        _cov2_counts = np.where(self.covs_2 == 0, 0, 1)
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
        resize: bool = False,
        **_,
    ) -> Union[str, PIL.Image.Image]:
        stack = PIL.Image.new("RGB", (c1.shape[1], c1.shape[0] + c2.shape[0]))

        rgb1 = np.dstack((c1, s1, np.zeros_like(c1)))
        img1 = PIL.Image.fromarray(rgb1, "RGB")

        rgb2 = np.dstack((c2, s2, np.zeros_like(c2)))
        img2 = PIL.Image.fromarray(rgb2, "RGB")

        stack.paste(img1, (0, 0))
        stack.paste(img2, (0, c1.shape[0]))

        if resize:
            stack = stack.resize(RESIZE_SIZE["squishstack"][::-1])

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
        resize: bool = False,
        ensure_JPGE_size: bool = True,
        **_,
    ) -> Union[str, PIL.Image.Image]:
        _c1, _s1, _c2, _s2 = Event2sampleReads.pad(c1, s1, c2, s2)
        cmyk = np.dstack((_c1, _s1, _c2, _s2))
        img = PIL.Image.fromarray(cmyk, "CMYK")

        if resize:
            img = img.resize(RESIZE_SIZE["squish4d"][::-1])

        if ensure_JPGE_size:
            while img.width * img.height > 65_500:  # max size for JPGE
                img = img.resize((img.width // 2, img.height // 2))

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


class Event2sampleComb(Event2sampleReads):
    def __init__(
        self,
        region: tuple,
        bam1: pysam.AlignmentFile,
        bam2: pysam.AlignmentFile,
        max_reads: int = -1,
    ) -> None:
        super().__init__(region, bam1, bam2, max_reads)

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

        if type == "comb":
            __plot = Event2sampleComb._plot2d
            ext = "png"
        elif type == "comb4d":
            __plot = Event2sampleComb._plot4d
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
            __ret.append(__plot(_c1, _s1, _c2, _s2, out=fout, mode=mode, **kwargs))

        return __ret

    @staticmethod
    def _plot2d(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
        resize: bool = False,
        **_,
    ) -> Union[str, PIL.Image.Image]:
        mc1, ms1, mc2, ms2 = [np.count_nonzero(x, axis=0) for x in [c1, s1, c2, s2]]
        h = max(mc1.max(), mc2.max(), ms1.max(), ms2.max())
        mc1mat = _getmat(mc1, h)
        ms1mat = _getmat(ms1, h)
        mc2mat = _getmat(mc2, h)
        ms2mat = _getmat(ms2, h)

        plp1 = np.dstack((mc1mat, ms1mat, np.zeros_like(mc1mat)))
        img1 = PIL.Image.fromarray(plp1, "RGB")
        plp2 = np.dstack((mc2mat, ms2mat, np.zeros_like(mc2mat)))
        img3 = PIL.Image.fromarray(plp2, "RGB")

        squish1 = np.dstack((c1, s1, np.zeros_like(c1)))
        img2 = PIL.Image.fromarray(squish1, "RGB")

        squish2 = np.dstack((c2, s2, np.zeros_like(c2)))
        img4 = PIL.Image.fromarray(squish2, "RGB")

        stack = PIL.Image.new(
            "RGB",
            (img1.width, img1.height + img2.height + img3.height + img4.height),
        )
        stack.paste(img1, (0, 0))
        stack.paste(img2, (0, img1.height))
        stack.paste(img3, (0, img1.height + img2.height))
        stack.paste(img4, (0, img1.height + img2.height + img3.height))

        if resize:
            stack = stack.resize(RESIZE_SIZE["comb"][::-1])

        if mode == "savefig":
            stack.save(out)
            return out
        elif mode == "pil":
            return stack
        else:
            raise ValueError

    @staticmethod
    def _plot4d(
        c1: np.ndarray,
        s1: np.ndarray,
        c2: np.ndarray,
        s2: np.ndarray,
        out: str,
        mode: str,
        resize: bool = False,
        ensure_JPGE_size: bool = True,
        **_,
    ) -> Union[str, PIL.Image.Image]:
        pileup = Event2sample._colmplp(
            *[np.count_nonzero(x, axis=0) for x in [c1, s1, c2, s2]],
            out="",
            mode="pil",
            ensure_JPGE_size=False,
        )
        squish = Event2sampleReads._squished_4d(
            c1, s1, c2, s2, out="", mode="pil", ensure_JPGE_size=False
        )

        stack = PIL.Image.new(
            "CMYK",
            (pileup.width, pileup.height + squish.height),
        )
        stack.paste(pileup, (0, 0))
        stack.paste(squish, (0, pileup.height))

        if resize:
            stack = stack.resize(RESIZE_SIZE["comb4d"][::-1])
        if ensure_JPGE_size:
            while stack.width * stack.height > 65_500:  # max size for JPGE
                stack = stack.resize((stack.width // 2, stack.height // 2))

        if mode == "savefig":
            stack.save(out)
            return out
        elif mode == "pil":
            return stack
        else:
            raise ValueError


def to_ranges(iterable: Iterable) -> Iterable:
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


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
        # print(_rows, subsample)
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


def main(args):
    CROSSFEEDS = [(0, 0), (0, 0.2), (0.2, 0), (0.2, 0.2)]

    __ix = 0
    _csvin = None

    _logfile = None
    if args.log:
        _logfile = open(args.log, "w")

    outcsv = open(args.outcsv, "w")

    with open(args.input, "r") as fin:
        _csvin = fin.readlines()

    for line in track(_csvin):
        __ix += 1
        if _logfile:
            _logfile.write(line)
        line = line.strip()

        if args.mode in ["pyplot2s", "colmplp2s"]:
            _region, _bam1, _bam2, _event = line.split(",")
            # print(_bam1, _bam2, _event)

            region = parse_region(_region)
            bam1 = pysam.AlignmentFile(pjoin(args.datadir, _bam1), "rb")
            bam2 = pysam.AlignmentFile(pjoin(args.datadir, _bam2), "rb")
            event = Event2sample(region, bam1, bam2, mincov=args.mincov)

            event.strip()

            bam1.close()
            bam2.close()

            prefix = f"{_region.replace(':','_')}.{_bam1.split('/')[-1].split('.')[1]}.{_bam2.split('/')[-1].split('.')[1]}"

            for _zoom in args.zoom:
                out_files = event.plot(
                    _zoom,
                    type=args.mode,
                    mode="savefig",
                    prefix=pjoin(args.outdir, f"zoom{_zoom}", prefix),
                    crossfeeds=CROSSFEEDS,
                    resize=args.resize,
                )
                for of in out_files:
                    outcsv.write(f"{basename(of)},{_event}\n")
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

            event.strip()

            bam1.close()
            bam2.close()

            prefix = f"{_region.replace(':','_')}.{_bam1.split('/')[-1].split('.')[1]}.{_bam2.split('/')[-1].split('.')[1]}"

            for _zoom in args.zoom:
                out_files = event.plot(
                    _zoom,
                    type=args.mode,
                    mode="savefig",
                    prefix=pjoin(args.outdir, f"zoom{_zoom}", prefix),
                    crossfeeds=CROSSFEEDS,
                    resize=args.resize,
                )
                for of in out_files:
                    outcsv.write(f"{basename(of)},{_event}\n")
                outcsv.flush()
        elif args.mode in ["comb", "comb4d"]:
            _region, _bam1, _bam2, _event = line.split(",")
            # print(_bam1, _bam2, _event)

            region = parse_region(_region)
            bam1 = pysam.AlignmentFile(pjoin(args.datadir, _bam1), "rb")
            bam2 = pysam.AlignmentFile(pjoin(args.datadir, _bam2), "rb")
            event = Event2sampleComb(region, bam1, bam2, max_reads=-1)  # TODO: fix this

            event.strip()

            bam1.close()
            bam2.close()

            prefix = f"{_region.replace(':','_')}.{_bam1.split('/')[-1].split('.')[1]}.{_bam2.split('/')[-1].split('.')[1]}"

            for _zoom in args.zoom:
                out_files = event.plot(
                    _zoom,
                    type=args.mode,
                    mode="savefig",
                    prefix=pjoin(args.outdir, f"zoom{_zoom}", prefix),
                    crossfeeds=CROSSFEEDS,
                    resize=args.resize,
                )
                for of in out_files:
                    outcsv.write(f"{basename(of)},{_event}\n")
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
    parser.add_argument(
        "-o", "--outdir", type=str, help="output directory", required=True
    )
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
        default="pyplot2s",
        choices=["pyplot2s", "colmplp2s", "squishstack", "squish4d", "comb", "comb4d"],
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
    parser.add_argument(
        "--resize",
        default=False,
        action="store_true",
        help="Resize images in output",
    )
    args = parser.parse_args()

    for z in args.zoom:
        makedirs(pjoin(args.outdir, f"zoom{z}"), exist_ok=True)

    main(args)
