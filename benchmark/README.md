# Manually curated benchmark

We considered the dataset provided in Best, A., James, K., Dalgliesh, C. et al. Human Tra2 proteins jointly control a CHEK1 splicing switch among alternative and constitutive target exons. Nat Commun 5, 4760 (2014). https://doi.org/10.1038/ncomms5760
[SRA BioProject ID: `PRJNA255099`]

To build a trustworthy truth set, we relied on two state-of-the-art tools for the differential quantification of AS events, namely [rMATS](https://doi.org/10.1073/pnas.1419161111) and [SUPPA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1417-1). 

We considered the alternative splicing events reported by both tools and 
we compare the two sets of significant events (i.e., those events with reported p-value < 0.05). 
Any event reported as significant by both tools is then considered a potential true candidate and added to the truth set.

A total of 62 alternative acceptor (A3), 54 alternative donor (A5), and 41 intron retention events resulted from this analysis. 
Although this truth set is assembled using the consensus of two accurate tools
we manually inspected their classification and found several regions
with debatable callings.

We thus provide the lists of manually curated events in the files:
- [Exon Skipping](ES.txt) 65 events
- [Alternative acceptor](A3.txt) 28 events
- [Alternative donor](A5.txt) 30 events
- [Intron retention](IR.txt) 17 events

Lastly we provide the bam files of the aligned reads from `PRJNA255099`, in which only the reads overlapping the region of interested are retained,
divided in two conditions:
- [Condition 1](c1.bam) comprised of samples `SRR1513329`, `SRR1513330`, `SRR1513331`
- [Condition 2](c2.bam) comprised of samples `SRR1513332`, `SRR1513333`, `SRR1513334`