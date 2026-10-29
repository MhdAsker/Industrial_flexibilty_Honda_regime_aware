# Honda H2 source acquisition

## Scope and result

H2 used only official Dryad and associated Scientific Data/PMC material. The 103,090,137,247-byte `data.zip` was not downloaded. Dryad exposes the small metadata objects independently, so `issues.zip` and the release `README.md` were acquired without bulk measurements. The existing H1 `reduced_data.zip` remained unchanged.

The exact machine-readable inventory is `results/provenance/source_acquisition_inventory.csv`. All acquired source objects are preserved under `data/raw/honda/provenance/`; meter-level data would be placed under `data/raw/honda/selective_full/`, which remains empty.

## Official Dryad objects

| Object | Bytes | Independently downloadable | Required in H2 | Status | SHA-256 | Purpose / H1 relationship |
|---|---:|---|---|---|---|---|
| `data.zip` (file 3910978) | 103,090,137,247 | Yes, but monolithic | No, without authorization | Not acquired | `a81cb72e4a9f5a2c8c0f10c2015bdc4b84f1945e605d5b9f7887ebeabcd711d2` | Contains meter-level raw/intermediate series needed for numerical source validation. |
| `issues.zip` (file 3910975) | 2,825,237 | Yes | Yes | Acquired and verified | `594e09131fc797af0a0cfb4355fd2dabe95d88a72fd78e73dbb3017240452a85` | Supplies exact issue intervals and correction method labels missing from H1 reduced data. MD5: `79e45712c8d3f7502a7ce0a519546d1f`. |
| `issues.zip` (file 3910976) | 2,825,237 | Yes | No | Duplicate listing; not downloaded twice | `594e09131fc797af0a0cfb4355fd2dabe95d88a72fd78e73dbb3017240452a85` | Same filename, size, and digest as file 3910975. |
| `plots.zip` (file 3910977) | 1,430,037 | Yes | No | Not acquired | `bd1d095b45d24a2c9e93dd6d5acf625b8f494fe6ead892df106711bfe4ab8591` | Publication plots are not needed to construct provenance masks. |
| `README.md` (file 3910982) | 15,222 | Yes | Yes | Acquired and verified | `7433ae4227aa9253c3a4c3f007486f5567d7fec2fdcf33b1600799217ff7cdee` | Defines release contents, processing levels, units, signs, and aggregates. |
| `reduced_data.zip` (file 3910979) | 320,157,291 | Yes | Yes | Existing H1 source, reverified | `2a35a629fdf863793ab1a0c1bf80110ec7fc909ea65f716613a529da4539571f` | Values preserved in the H2 SILVER layer. |

Dryad DOI: `10.5061/dryad.73n5tb363`.

## Associated official documentation

| Object | Bytes | Status | MD5 | SHA-256 |
|---|---:|---|---|---|
| Scientific Data article HTML from PMC | 234,487 | Acquired over HTTPS | `93948ef151cda6a3e002ecc1020c3ee9` | `af05da064d4c4d4fb5996df5844dcdfff09da02a432b0922581b79b324062dc5` |
| Electrical hierarchy Figure 1 | 202,146 | Acquired over HTTPS | `55efb32c45362c6673a71c8603155b0d` | `502dbba656caf3d312ee74ebb27aa0df7b9d409b3d64e4922b3884d7426c8a43` |
| Thermal hierarchy Figure 2 | 48,504 | Acquired over HTTPS | `c842a427b1e2b7e34bc8388bc23266a8` | `a127ad16f215ce5687c99941a7969560e059fe4808c2f6dce2e94c209c8b7b94` |

Associated paper DOI: `10.1038/s41597-025-05186-3`; preserved article source: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12103535/`.

## Packaging consequence

Metadata acquisition is selective, but measurement acquisition is not: Dryad places the raw, harmonized, corrected, and resampled meter-level files together in the 103 GB `data.zip`. Consequently H2 can recover issue timing, processing semantics, hierarchy, and aggregate membership without that archive, but cannot independently compare meter-level values with reduced aggregates or validate the gross-load balance. No automatic escalation to the full archive occurred.
