from os.path import join as pjoin
from sklearn.model_selection import train_test_split
import pandas as pd


PLOT="plot2s.py"
DS="deepSpecas.py"
DSESM="dsEnsemble.py"

SIMDIR="/data/deepSpecas/simulations/simulated_reads/"
INPUTCSV="/data/deepSpecas/simulations/events.2s.csv"

PLOTOUT="/data/deepSpecas/images/"
NETOUT="/data/deepSpecas/net"

BENCHMARKCSV="/data/deepSpecas/benchmark/"
BENCHMARKOUT="/data/deepSpecas/predict_benchmark/"

ZOOMS=[-1, 20]
# ZOOMS=[20]

rule all:
    input:
        expand(
            # pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "events.csv"),
            pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
            # pjoin(BENCHMARKOUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}", "prediction.{bevent}.csv"),
            mincov=[3],
            mode=["pyplot2s","colmplp2s", "squishstack", "squish4d", "comb", "comb4d"],
            version=[1],
            zoom=ZOOMS,
            # --- net ---------
            epochs=[10],
            cse=["Y", "N"],
            a=["Y", "N"],
            size = [0.5],
            net = ["rs50"],
            samplefrac = [0.5],
            nocf = ["_", "0.2-0.2"],
            # bevent = ["cse"],
        )

rule benchmark:
    input:
        expand(
            pjoin(BENCHMARKOUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}", "prediction.{bevent}.csv"),
            mincov=[3],
            version=[1],
            zoom=ZOOMS,
            mode=["pyplot2s","colmplp2s", "squishstack", "squish4d", "comb", "comb4d", "ensemble"],
            # --- net ---------
            epochs=[10],
            cse=["Y", "N"],
            a=["Y", "N"],
            size = [0.5],
            net = ["rs50"],
            samplefrac = [0.5],
            nocf = ["_", "0.2-0.2"],
            bevent = ["cse", "a3", "a5", "ri"],
        )

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
        --datadir {SIMDIR} \
        --outcsv {output.csv} \
        -o {params.outdir} \
        --log {log.pltlog} \
        --mincov {wildcards.mincov} \
        --mode {wildcards.mode} \
        --zoom {params.zooms} \
        --resize \
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
        net=pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
        hm=pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.hm.png"),
        log=pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.log"),
    params:
        wd=pjoin(PLOTOUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}"),
        batchsize=8,
        cse = lambda wildcards: '--cse' if wildcards.cse == "Y" else '',
        a = lambda wildcards: '--a' if wildcards.a == "Y" else '',
        nocf = lambda wildcards: "_" if wildcards.nocf == "_" else wildcards.nocf,
    resources:
        gpu=1
    shell: 
        """
            python {DS} train \
            --train {input.train} \
            --val {input.val} \
            --wd {params.wd} \
            --hm {output.hm} \
            -o {output.net} \
            --epochs {wildcards.epochs} \
            --batch-size {params.batchsize} \
            --plot {wildcards.mode} \
            --plotversion {wildcards.version} \
            --sample-frac {wildcards.samplefrac} \
            --no-cf {params.nocf} \
            --resizescale {wildcards.size} {params.cse} {params.a} \
            > {output.log}
        """

rule predict_benchmark:
    input: 
        csv = pjoin(BENCHMARKCSV, "events.{bevent}.csv"),
        net = pjoin(NETOUT, "mincov{mincov}", "{mode}", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
    output: 
        out = pjoin(BENCHMARKOUT, "mincov{mincov}", "{mode,pyplot2s|colmplp2s|squishstack|squish4d|comb|comb4d}", "v{version}", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}", "prediction.{bevent}.csv"),
    params:
        outdir = pjoin(BENCHMARKOUT, "mincov{mincov}", "{mode}", "v{version}", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}"),
        cse = lambda wildcards: "--cse" if wildcards.cse == "Y" else '',
        a = lambda wildcards: "--a" if wildcards.a == "Y" else '',
    resources:
        gpu=1
    shell:
        """
        python {DS} predict --csv {input.csv} \
            -n {input.net} \
            --plot {wildcards.mode} \
            --zoom {wildcards.zoom} \
            {params.cse} {params.a} \
            > {output.out}
        """

rule ensemble_benchmark:
    input:
        csv = pjoin(BENCHMARKCSV, "events.{bevent}.csv"),
        net_pyplot2s = pjoin(NETOUT, "mincov{mincov}", "pyplot2s", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
        net_colmplp2s = pjoin(NETOUT, "mincov{mincov}", "colmplp2s", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
        net_squishstack = pjoin(NETOUT, "mincov{mincov}", "squishstack", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
        net_squish4d = pjoin(NETOUT, "mincov{mincov}", "squish4d", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
        net_comb = pjoin(NETOUT, "mincov{mincov}", "comb", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
        net_comb4d = pjoin(NETOUT, "mincov{mincov}", "comb4d", "v{version}", "train", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}.pth"),
    output:
        out = pjoin(BENCHMARKOUT, "mincov{mincov}", "ensemble", "v{version}", "zoom{zoom}", "{net}.cse{cse}.a{a}.e{epochs}.resize{size}.nocf{nocf}.sf{samplefrac}", "prediction.{bevent}.csv"),
    params:
        cse = lambda wildcards: "--cse" if wildcards.cse == "Y" else '',
        a = lambda wildcards: "--a" if wildcards.a == "Y" else '',
    resources:
        gpu=1
    shell:
        """
        python {DSESM} --csv {input.csv} \
        --pyplot2s {input.net_pyplot2s} \
        --colmplp2s {input.net_colmplp2s} \
        --squishstack {input.net_squishstack} \
        --squish4d {input.net_squish4d} \
        --comb {input.net_comb} \
        --comb4d {input.net_comb4d} \
        --zoom {wildcards.zoom} \
        {params.cse} {params.a} \
        > {output.out}
        
        """
