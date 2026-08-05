# RiverBench

Two **temporal data lakes** — collections of tables tracked across their versions — together with the full pipelines that build them. Both lakes label every table's *versions* and their *lineage* (which edition precedes which), which is what makes them usable for temporally-valid table search and for studying schema and content drift.

The two lakes are deliberately complementary:

| | [Wikipedia Tables](wikipedia-tables/) | [NYC Open Data](nyc-open-data/) |
|---|---|---|
| Authors | Many contributors and bots | A few data stewards |
| Updates | Frequent, unstructured | Infrequent, governed |
| Version labels | Free — from the page revision history | Latent — must be reconstructed, then curated by hand |
| Language | Go | Python |

- **[wikipedia-tables/](wikipedia-tables/)** — extracts and tracks every table across every revision of every article in the English Wikipedia XML dumps (2001 to March 2026). Because the revision history *is* the version history, the version and lineage labels are gold-standard by construction. See [wikipedia-tables/README.md](wikipedia-tables/README.md).

- **[nyc-open-data/](nyc-open-data/)** — scrapes every table dataset on [data.cityofnewyork.us](https://data.cityofnewyork.us) and reconstructs the temporal structure the portal never records. Here each edition is an independent dataset with no link to its predecessors, so versions and lineage are recovered from several complementary signals and then reviewed by human annotators. See [nyc-open-data/README.md](nyc-open-data/README.md).

## Data
## Will be added after the revision process to remain anonymous here: https://doi.org/10.20383/103.01750
### Creating an anonymous link to the data with such size was not possible.

The generated lakes are published on Zenodo:

**[Zenodo record — placeholder]()** *(link to be added)*

The repository holds the pipelines, not the data. Running either pipeline reproduces its lake from the public source.

## Technical Report

Design decisions, corpus statistics, and the phenomena each pipeline has to handle are described in:

**[Technical report — placeholder]()** *(link to be added)*

## Repository layout

```
wikipedia-tables/    Go module: dump -> JSONL of table version histories
nyc-open-data/       Python package: portal -> per-dataset folders + version series + lineage
```

Each folder is self-contained and documents its own build, usage, and output format.
