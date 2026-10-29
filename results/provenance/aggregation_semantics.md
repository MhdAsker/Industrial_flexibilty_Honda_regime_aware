# Fifteen-minute aggregation semantics

The associated paper states that issue correction and clock alignment occur before one-minute resampling. The February 2025 release then forms non-cumulative 15-minute values as arithmetic means. H1 independently confirmed left-closed, left-labelled UTC arithmetic means from the released one-minute data.

H2 could not independently reproduce raw/intermediate → corrected one-minute → reduced 15-minute processing because all meter-level series are inside the 103 GB monolithic `data.zip`. Issue intervals are mapped to every 15-minute bin whose half-open interval overlaps the inclusive issue interval. Pre-April-2020 Tixi-fed grid/CHP values are conservatively classified as deterministic clock corrections because the exact March synchronization day is not documented. Therefore downsampling semantics are confirmed, while full raw-to-reduced semantics are not.
