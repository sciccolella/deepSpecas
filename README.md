# Build env

```bash
mamba create -n DSpoly2 -c bioconda -c conda-forge -c pytorch -c nvidia pysam snakemake matplotlib rich pytorch torchvision  torchaudio pytorch-cuda=11.8 scikit-learn pandas seaborn opencv
conda activate DSpoly2
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```


# Run `deepSpecas` using trained nets

Alongside the tool we also release the trained networks that we used in the manuscript, therefore allowing the use to use `deepSpecas` directly without the need of retrain all the different implementations.

## Ensemble
The help of the tool:
```
usage: dsEnsemble.py [-h] [--zoom ZOOM [ZOOM ...]] [--version VERSION] --csv CSV [--explain] [--imgout IMGOUT] [--mincov MINCOV] [--pyplot2s PYPLOT2S] [--colmplp2s COLMPLP2S] [--squishstack SQUISHSTACK] [--squish4d SQUISH4D] [--comb COMB] [--comb4d COMB4D] [--a] [--cse]

Predict using ensemble of nets

options:
usage: dsEnsemble.py [-h] [--zoom ZOOM] [--version VERSION] --csv CSV [--explain] [--imgout IMGOUT] [--mincov MINCOV] [--pyplot2s PYPLOT2S] [--colmplp2s COLMPLP2S] [--squishstack SQUISHSTACK] [--squish4d SQUISH4D] [--comb COMB] [--comb4d COMB4D] [--a] [--cse]

Predict using ensemble fo nets

options:
  -h, --help            show this help message and exit
  --zoom ZOOM           zoom
  --version VERSION     Plotting version for each mode [default=1]
  --csv CSV             CSV to predict [region,BAM_path,GTF_path]
  --explain             Produce CAM on classification [DEFAULT: False]
  --imgout IMGOUT       Output dir for saving images [DEFAULT: False]
  --mincov MINCOV       minimum covarege to be plot [default=0]
  --pyplot2s PYPLOT2S   Trained model on pyplot2s
  --colmplp2s COLMPLP2S
                        Trained model on colmplp2s
  --squishstack SQUISHSTACK
                        Trained model on squishstack
  --squish4d SQUISH4D   Trained model on squish4d
  --comb COMB           Trained model on comb
  --comb4d COMB4D       Trained model on comb4d
  --a                   Collapse A3 and A5 into A. [DEFAULT: False]
  --cse                 Collapse CE and SE into CSE. [DEFAULT: False]
```

Run example, the CSV file one whishes to use needs to be of the following format:

```
Genomic_Postions,Path_to_BAM_for_condition1,Path_to_BAM_for_condition2
```
We leave as example the files in `benchmark/[A3|A5|ES|IR].csv`

Example run on A3 for benchmark dataset:
```bash
python dsEnsemble.py \
--csv benchmark/A3.csv \
--pyplot2s data/trained_nets/pyplot2s/trained.pth \
--colmplp2s data/trained_nets/colmplp2s/trained.pth \
--squishstack data/trained_nets/squishstack/trained.pth \
--squish4d data/trained_nets/squish4d/trained.pth \
--comb data/trained_nets/comb/trained.pth \
--comb4d data/trained_nets/comb4d/trained.pth \
--zoom -1 --a --cse
```

## Single representation

To run on a single image representation it is necessary to run the tool `deepSpecas.py` in `predict` mode, of which the help is shown here:
```
usage: deepSpecas.py predict [-h] --csv CSV [--explain] [--imgout IMGOUT] --zoom ZOOM [--mincov MINCOV] -n NET [-m {rs50}] [--wd WD] [--a] [--cse] [--resizescale RESIZESCALE] -p {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d} [--plotversion PLOTVERSION]

options:
  -h, --help            show this help message and exit
  --csv CSV             CSV to predict [region,BAM_path,GTF_path]
  --explain             Produce CAM on classification [DEFAULT: False]
  --imgout IMGOUT       Output dir for saving images [DEFAULT: False]
  --zoom ZOOM           Zoom
  --mincov MINCOV       minimum covarege to be plot [default=0]
  -n NET, --net NET     Trained net file
  -m {rs50}, --model {rs50}
                        Network model
  --wd WD               Working directory for filenames in CSV. [DEFAULT: '']
  --a                   Collapse A3 and A5 into A. [DEFAULT: False]
  --cse                 Collapse CE and SE into CSE. [DEFAULT: False]
  --resizescale RESIZESCALE
                        Resize images to desired rescaled. [DEFAULT: 1.0]
  -p {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}, --plot {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}
                        Plot mode
  --plotversion PLOTVERSION
                        Plot version
```

Example run on Intron retention for benchmark dataset using `squishplot` representation:
```bash
python deepSpecas.py predict --csv benchmark/IR.csv --zoom -1 -n data/trained_nets/squishstack/trained.pth --a --cse -p squishstack
```

# Train `deepSpecas`

## Build images from samples
To build the image representation from samples it is necessary to run the tool `plot2s.py`, usage:
```
usage: plot2s.py [-h] -i INPUT -o OUTDIR [-d DATADIR] --outcsv OUTCSV [-m {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}] [--mincov MINCOV] [--zoom [ZOOM ...]] [--version VERSION] [--log LOG] [--resize]

Plot figures from mpileup

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        CSV file of events
  -o OUTDIR, --outdir OUTDIR
                        output directory
  -d DATADIR, --datadir DATADIR
                        root directory containing bams and gtfs
  --outcsv OUTCSV       Output CSV file of events
  -m {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}, --mode {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}
                        Plot type
  --mincov MINCOV       minimum covarege to be plot [default=0]
  --zoom [ZOOM ...]     zoom [default=-1]
  --version VERSION     Plotting version for each mode [default=1]
  --log LOG             Log output file
  --resize              Resize images in output
```

## Train the nets
To run on a single image representation it is necessary to run the tool `deepSpecas.py` in `train` mode, of which the help is shown here:
```
usage: deepSpecas.py train [-h] --train TRAIN [--val VAL] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--crossvalidation] [--sample-frac SAMPLE_FRAC] [--no-cf NO_CF] [-o NET_OUT] [-m {rs50,TODO}] [--wd WD] [--a] [--cse] [--resizescale RESIZESCALE] -p
                           {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d} [--plotversion PLOTVERSION] [--hm HM]

options:
  -h, --help            show this help message and exit
  --train TRAIN         CSV file
  --val VAL             CSV file. If not provided `train` will be randomly split
  --epochs EPOCHS       Training epochs
  --batch-size BATCH_SIZE
                        Batch size
  --crossvalidation     Perform cross-validation instead of training on all dataset
  --sample-frac SAMPLE_FRAC
                        Downsample input when training [Default: 1]
  --no-cf NO_CF         Ignore a crossfeed value [Default: '_']
  -o NET_OUT, --net-out NET_OUT
                        Output file for trained network
  -m {rs50,TODO}, --model {rs50,TODO}
                        Network model
  --wd WD               Working directory for filenames in CSV. [DEFAULT: '']
  --a                   Collapse A3 and A5 into A. [DEFAULT: False]
  --cse                 Collapse CE and SE into CSE. [DEFAULT: False]
  --resizescale RESIZESCALE
                        Resize images to desired rescaled. [DEFAULT: 1.0]
  -p {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}, --plot {pyplot2s,colmplp2s,squishstack,squish4d,comb,comb4d}
                        Plot mode
  --plotversion PLOTVERSION
                        Plot version
  --hm HM               Heatmap out [default=hm.png]
  ```

# Replicate experiments
The entire pipeline is available as a [snakefile](runDS.smk).