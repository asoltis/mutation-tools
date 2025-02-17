# Structural variant analysis tools

SV clustering:
--------------

Scripts for clustering single and multi-sample SV calls (clustering/ folder).
As implemented in [Soltis et al. Cell Rep Med 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9729884/pdf/main.pdf):
    - Methods described in "Somatic structural variant clustering" section
    - Results from procedures as displayed in supplementary Figure S4C

Scripts:
    - cluster_individual_sample_SV_chains.py
        - clusters single sample SV calls, allowing for "chains" of SVs
        - Creates a breakend graph where an edge between two SVs is created if either of the two ends intersect within a threshold Dt
        - Clusters are reported as connected component graphs from the full graph

    - cluster_multisample_SVs_chains.py
        - clusters multisample SV calls, allowing for "chains" of SVs
        - This script utilizes similar methods to what is described above for single samples

    - cluster_multisample_SV_events.py
        - clusters multisample SV calls, but attempts to find similar "events" by start and end coordinates
        - Intrachromosomal SVs (deletions, tandem duplications, insertions, and inversions) are treated distinctly from interchromosomal events
        - Script first finds raw overlapping and closely neighboring events, then applies single linkage clustering based on an event
          distance function and a threshold Ct for final clusters.


