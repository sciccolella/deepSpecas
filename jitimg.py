import plot as DSplot
import pysam
import numpy as np
import DSutils


def get_img(event_data: tuple[np.array], mode: str):
    if mode == "pyplot":
        return DSplot.pyplot_plt(event_data, out=None, mode="pil")
    elif mode == "colmplp":
        return DSplot.colmplp_plt(event_data, out=None, mode="pil")
    else:
        raise ValueError


def get_event(region: str, bampath: str, gtfpath: str, mincov: int = 0) -> DSplot.Event:
    _r = DSplot.parse_region(region)
    bam = pysam.AlignmentFile(bampath, "rb")
    _exons = DSutils.parse_gtf(gtfpath)
    transcript = DSutils.get_transcript(_exons, _r[1], _r[2])
    event = DSplot.Event(_r, bam, transcript, mincov=mincov)
    bam.close()
    return event


def get_multi_events(
    region: str, bampath: str, gtfpath: str, mincov: int = 0
) -> dict[str, DSplot.Event]:
    me = dict()
    _r = DSplot.parse_region(region)
    bam = pysam.AlignmentFile(bampath, "rb")
    _exons = DSutils.parse_multi_gtf(gtfpath)
    for kt in _exons:
        t = DSutils.get_transcript(_exons[kt], _r[1], _r[2])
        e = DSplot.Event(_r, bam, t, mincov=mincov)
        me[kt] = e

    bam.close()
    return me


def get_event2s(region: str, bam1_path: str, bam2_path: str, plot:str, mincov: int = 0, max_reads: int = -1) -> DSplot.Event2sample:
    _r = DSplot.parse_region(region)
    bam1 = pysam.AlignmentFile(bam1_path, "rb")
    bam2 = pysam.AlignmentFile(bam2_path, "rb")
    if plot in ["pyplot2s","colmplp2s"]:
        event = DSplot.Event2sample(_r, bam1, bam2, mincov=mincov)
    elif plot in ["squishstack", "squish4d"]:
        event = DSplot.Event2sampleReads(_r, bam1, bam2, max_reads=max_reads)
    else:
        raise ValueError

    bam1.close()
    bam2.close()

    return event


if __name__ == "__main__":
    # TESTING OF THE FILE
    region = "1:55058597-55061425"
    bampath = "/home/simone/data/benchmark/1/SRR32.bam"
    gtfpath = "/home/simone/data/benchmark/1/transcripts/ENSG00000169174.gtf"
    mode = "pyplot"
    mincov = 3
    zooms = [-1, 20]
    e = get_event(region, bampath, gtfpath, mincov=mincov)
    e.strip()
    for i, z in enumerate(zooms):
        e.build_zoom(z)
        img = get_img(e.zoom[z], mode)
        img.save(f"/home/simone/code/specas/jit.test.z{z}.png")
