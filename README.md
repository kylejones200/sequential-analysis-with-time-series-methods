# Sequential Analysis with Time Series Methods

Companion code for the Medium article [Sequential Analysis with Time Series Methods](https://medium.com/@kyle-t-jones/sequential-analysis-with-time-series-methods-7c15b76576a7).

## Business context

While time series analysis is traditionally applied to temporal data, its principles and methods extend naturally to sequential data — data where order matters but isn't strictly tied to time. These sequences include human poses (yoga, martial arts, or golf swings), spectroscopic data (e.g., wine analysis), or even patterns in chess moves.

Sequential data represents ordered observations where the sequence carries significant meaning. Unlike traditional time series, these sequences may not correspond to a specific timestamp but instead reflect progression through a series of states or measurements.

1. **Order dependency:** Each observation is influenced by its position in the sequence.
2. **Cyclic patterns:** Some sequences exhibit recurring patterns (e.g., yoga flows or periodic spectral peaks in wine analysis).
3. **Multidimensionality:** Sequences often involve multiple variables evolving together, such as joint angles in a golf swing or RGB spectra in wine analysis.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kylejones200/sequential-analysis-with-time-series-methods.git
cd sequential-analysis-with-time-series-methods
uv sync
uv run pytest
uv run python sequential_analysis_time_series.py
```

The main script runs five sequential-analysis demos (golf swing DTW, wine spectra, chess moves, protein structure, industrial anomaly detection) and writes `sequential_*.png` figures.

The shorter draft companion from the article lives in `draft_Sequential-Analysis-with-Time-Series-Methods-7c15b76576a7.py`.

## Repository layout

| File | Purpose |
| --- | --- |
| `sequential_analysis_time_series.py` | Full demo script with five domain examples |
| `draft_Sequential-Analysis-with-Time-Series-Methods-7c15b76576a7.py` | Article companion (golf smoothing + wine/chess snippets) |
| `config.yaml` | Seeds, model defaults, and figure settings |
| `article.md` | Article text excerpt |
| `Sequential-Analysis-with-Time-Series-Methods.md` | Full article draft |

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
