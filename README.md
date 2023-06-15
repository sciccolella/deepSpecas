# Build env

```bash
mamba create -n DSpoly2 -c bioconda -c conda-forge -c pytorch -c nvidia pysam snakemake matplotlib rich pytorch torchvision  torchaudio pytorch-cuda=11.8 scikit-learn pandas seaborn opencv
conda activate DSpoly2
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```


# Run predict

python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/ -m rs50 -p pyplot2s
python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/pyplot2s/v1/train/zoom20/rs50.cA.e20.full.nocf0.2-0.2.sf0.5.pth -m rs50 -p pyplot2s > debug/prediction/benchmark/2samples/1/pyplot2s.rs50.cA.e20.full.nocf0.2-0.2.sf0.5.txt

python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/ -m rs50 -p colmplp2s
python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n home/simone/polyDS/net/mincov3/colmplp2s/v1/train/zoom20/rs50.cA.e20.full.nocf0.2-0.2.sf0.5.pth -m rs50 -p colmplp2s > debug/prediction/benchmark/2samples/1/colmplp2s.rs50.cA.e20.full.no02-02.txt


python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/squish4d/v1/train/zoom20/rs50.cA.e20.full.nocf_.sf0.5.pth -m rs50 -p squish4d > debug/prediction/benchmark/2samples/1/squish4d.rs50.cA.e20.full.nocf_.sf0.5.txt
python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/squish4d/v1/train/zoom20/rs50.cA.e20.full.nocf0.2-0.2.sf0.5.pth -p squish4d > debug/prediction/benchmark/2samples/1/squish4d.rs50.cA.e20.full.nocf0.2-0.2.sf0.5.txt

python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/squishstack/v1/train/zoom20/rs50.cA.e20.full.nocf_.sf0.5.pth -m rs50 -p squishstack > debug/prediction/benchmark/2samples/1/squishstack.rs50.cA.e20.full.nocf_.sf0.5.txt
python poly/deepSpecas.py predict --csv /home/simone/data/benchmark/2samples/1/predict.csv -n /home/simone/polyDS/net/mincov3/squishstack/v1/train/zoom20/rs50.cA.e10.full.nocf0.2-0.2.sf0.5.hm.png -m rs50 -p squishstack > debug/prediction/benchmark/2samples/1/squishstack.rs50.cA.e20.full.nocf0.2-0.2.sf0.5.txt

# Degub
python poly/plot.py -i ~simone/data/polyester_sims/imgs/events.2s.csv --datadir ~simone/data/polyester_sims --mincov 0 --zoom 20 -o debug/squishstack/ --mode squishstack --outcsv debug/squishstack/events.csv

python poly/plot.py -i ~simone/data/polyester_sims/imgs/events.2s.csv --datadir ~simone/data/polyester_sims --mincov 0 --zoom 20 -o debug/squish4d/ --mode squish4d --outcsv debug/squish4d/events.csv

