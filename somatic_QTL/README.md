# Directory for somatic quantitative trait loci (QTL) analysis

QTL analysis estimates effect(s) of mutations/variants on expression of target genes/proteins. The scripts here perform analysis considering
coding gene mutations/alterations as well as non-coding mutations. 

As implemented in [Soltis et al. Cell Rep Med 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9729884/pdf/main.pdf):
- Methods described in "Somatic quantitative trait loci (QTL) analysis" section
- Results from procedures as displayed in supplementary Figure S4B
- "PEER" refers to probabilistic estimation of expression residuals as described [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000770)
- "VST" refers to variance stabililizing transformation

Scripts:

- somatic_eQTL_MutEnricher_coding_PEER_exp.py
    - Code to run somatic QTL analysis from MutEnricher coding gene analysis output and expression data with PEER estimated factors on 
      VST-normalized expression data (for hidden factor terms)
    - Additional inputs used for regression include: copy number levels, ancestry, sex
    - Several regression techniques are implemented: standard OLS, Ridge regression, Lasso regression, and Elastic Net

- somatic_eQTL_MutEnricher_noncoding_PEER_exp.py
    - Run somatic QTL analysis with MutEnricher noncoding analysis results. 
    - General procedure is the same as above, but noncoding regions are mapped to genes and handled by a few different grouping methods.

- plotting/
    - scripts for plotting individual results from above analyses

