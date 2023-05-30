import numpy as np
import re

def get_transcript(exons: list[tuple], start: int, end: int, fill: int = 1):
    _ix_ex = 0
    t = np.zeros(end - start, dtype=np.uint8)
    for pos in range(start, end):
        if _ix_ex >= len(exons):
            continue
        if pos >= exons[_ix_ex][0] and pos <= exons[_ix_ex][1]:
            t[pos - start] = fill
        if pos > exons[_ix_ex][1]:
            _ix_ex += 1
    return t

def parse_gtf(fpath: str) -> list[tuple]:
    exons = []
    for line in open(fpath):
        line = line.strip()
        type = line.split("\t")[2]
        if type == "exon" and "exon_number" in line:
            _data = line.split("\t")
            exons.append((int(_data[3]), int(_data[4])))

    return sorted(exons, key=lambda x: x[0])

def parse_multi_gtf(fpath: str) -> list[tuple]:
    exons = {}
    for line in open(fpath):
        line = line.strip()
        type = line.split("\t")[2]
        if type == "exon" and "exon_number" in line:
            _data = line.split("\t")
            info = _data[-1].split(" ")
            tid = info[info.index("transcript_id")+1][:-1].strip('"')
            if not tid in exons:
                exons[tid] = []
            exons[tid].append((int(_data[3]), int(_data[4])))

    for t in exons:
        exons[t] = sorted(exons[t], key=lambda x: x[0])
    return exons

if __name__ == "__main__":
    # TESTING THE FUNCTIONS
    multigtf = "debug/test.multi.gtf"

    x = parse_multi_gtf(multigtf)
    print(x)