from os.path import join as pjoin
from sklearn.model_selection import train_test_split
import pandas as pd


PLOT="poly/plot.py"
DS="poly/deepSpecas.py"

DATADIR="/home/simone/data/polyester_sims/"
# INPUTCSV="/home/simone/data/polyester_sims/imgs/events.csv"
INPUTCSV="/home/simone/data/polyester_sims/imgs/events.2s.csv"

PLOTOUT="/home/simone/polyDS/images/"
NETOUT="/home/simone/polyDS/net/"

BENCHMARKCSV="/home/simone/data/benchmark/2samples/1/predict.merged.csv"
PREDICTOUT="/home/simone/polyDS/predict_benchmark/"
POUT="2samples.1.merged"

# ZOOMS=[-1, 20]
ZOOMS=[20]

rule all:
    input:
        expand(
            # pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}.pth"),
            pjoin(PREDICTOUT, POUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}.txt"),
            mincov=[3],
            # mode=["pyplot", "colmplp"],
            mode=["pyplot2s","colmplp2s", "squishstack", "squish4d"],
            version=[1],
            zoom=ZOOMS,
            # --- net ---------
            epochs=[10],
            classes=["A", "CSE"],
            size = ["full"],
            net = ["rs50"],
            samplefrac = [0.5],
            nocf = ["_", "0.2-0.2"]
        ),

rule make_plots:
    input: 
        csv=INPUTCSV
    output: 
        csv=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.csv")
    params:
        outdir=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}"),
        zooms = " ".join(str(x) for x in ZOOMS),
    log:
        pltlog=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "plt.log"),
        stderr=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "err.log")
    shell: 
        """
        python {PLOT} -i {input.csv} \
        --datadir {DATADIR} \
        --outcsv {output.csv} \
        -o {params.outdir} \
        --log {log.pltlog} \
        --mincov {wildcards.mincov} \
        --mode {wildcards.mode} \
        --zoom {params.zooms} \
        2> {log.stderr}
        """

rule train_test_split_data:
    input: 
        csv=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.csv")
    output: 
        train=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.train.csv"),
        val=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.val.csv")
    run: 
        df = pd.read_csv(input.csv)
        train, test = train_test_split(df, test_size=0.3)
        train.to_csv(output.train, index=False, header=False)
        test.to_csv(output.val, index=False, header=False)

rule train_net:
    input: 
        train=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.train.csv"),
        val=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.val.csv")
    output: 
        net=pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}.pth"),
        hm=pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}.hm.png"),
    params:
        wd=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}"),
    run: 
        resize = '--resize' if wildcards.size == "resize" else ''
        collapse = '--collapse' if wildcards.classes == "CSE" else ''
        nocf = "_" if wildcards.nocf == "_" else wildcards.nocf 

        shell("""
            python {DS} train \
            --train {input.train} \
            --val {input.val} \
            --wd {params.wd} \
            --hm {output.hm} \
            -o {output.net} \
            --epochs {wildcards.epochs} \
            --plot {wildcards.mode} \
            --plotversion {wildcards.version} \
            --sample-frac {wildcards.samplefrac} \
            --no-cf {nocf} \
            {resize} {collapse}
        """)

rule predict_benchmark:
    input: 
        csv = BENCHMARKCSV,
        net=pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}.pth"),
    output: 
        out = pjoin(PREDICTOUT, POUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}.txt"),
    params:
        imgout = pjoin(PREDICTOUT, POUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}", "{net}.c{classes}.e{epochs}.{size}.nocf{nocf}.sf{samplefrac}"),
    run:
        collapse = '--collapse' if wildcards.classes == "CSE" else ''
        shell(
            """
            mkdir -p {params.imgout}
            python poly/deepSpecas.py predict --csv {input.csv} \
            -n {input.net} \
            --plot {wildcards.mode} \
            --imgout {params.imgout} --explain \
            {collapse} \
            --mincov 0 \
            > {output.out}
            """
        )