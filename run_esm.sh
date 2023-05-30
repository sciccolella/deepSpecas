#!/bin/bash

set -xe;

# for MINCOV in 0 3 
# do
#     python poly/esmPredict.py \
#     --csv  /home/simone/data/benchmark/2samples/2.A/3v3.A3.predict.csv \
#     --pyplot2s /home/simone/polyDS/net/mincov3/pyplot2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth \
#     --colmplp2s /home/simone/polyDS/net/mincov3/colmplp2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth \
#     --squishstack /home/simone/polyDS/net/mincov3/squishstack/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth \
#     --squish4d /home/simone/polyDS/net/mincov3/squish4d/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth \
#     --classes A3 A5 CSE RI NN \
#     --imgout debug/ensemble/a3/img.mincov$MINCOV \
#     --mincov $MINCOV > debug/ensemble/a3/prediction.mincov$MINCOV.csv
# done

# for MINCOV in 0 3 
# do
#     python poly/esmPredict.py \
#     --csv  /home/simone/data/benchmark/2samples/2.A/3v3.A5.predict.csv \
#     --pyplot2s /home/simone/polyDS/net/mincov3/pyplot2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth \
#     --colmplp2s /home/simone/polyDS/net/mincov3/colmplp2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth \
#     --squishstack /home/simone/polyDS/net/mincov3/squishstack/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth \
#     --squish4d /home/simone/polyDS/net/mincov3/squish4d/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth \
#     --classes A3 A5 CSE RI NN \
#     --imgout debug/ensemble/a5/img.mincov$MINCOV \
#     --mincov $MINCOV > debug/ensemble/a5/prediction.mincov$MINCOV.csv
# done

for MINCOV in 0 3 
do
    python poly/esmPredict.py \
    --csv  /home/simone/data/benchmark/2samples/2.A/3v3.RI.predict.csv \
    --pyplot2s /home/simone/polyDS/net/mincov3/pyplot2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth \
    --colmplp2s /home/simone/polyDS/net/mincov3/colmplp2s/v1/train/zoom20/rs50.cCSE.e10.full.nocf_.sf0.5.pth \
    --squishstack /home/simone/polyDS/net/mincov3/squishstack/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth \
    --squish4d /home/simone/polyDS/net/mincov3/squish4d/v1/train/zoom20/rs50.cCSE.e10.full.nocf0.2-0.2.sf0.5.pth \
    --classes A3 A5 CSE RI NN \
    --imgout debug/ensemble/ri/img.mincov$MINCOV \
    --mincov $MINCOV > debug/ensemble/ri/prediction.mincov$MINCOV.csv
done