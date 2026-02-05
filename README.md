# H1 Dyadic Law
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18491143.svg)](https://doi.org/10.5281/zenodo.18491143)

## Project Structure
H1_Dyadic_Simulation_Full_Repo
├── data
│   ├── README.md                          # Data documentation
│   └── WPP2024_TotalPopulationBySex.csv   # UN 2025 population data
├── deltaF_hist_generation.py              # Script to generate ΔF histograms
├── h1_dyadic_law-timed-precision.py       # Main simulation script
├── LICENSE                                # License (MIT recommended)
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── h1_v24.pdf                             # Compiled final paper
└── LaTeX                                  # Paper source files
├── deltaF_hist_example.pdf                # Example histogram figure
├── h1_v24.tex                             # Main LaTeX source
└── references.bib                         # BibTeX references

This layout separates:
- **data/** → raw input files
- **code** → simulation & visualization scripts
- **paper** → LaTeX source + compiled PDF

## How to Reproduce

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

Run simulation (example: 10,000 dyads)

Bash
python h1_dyadic_law-timed-precision.py --n_dyads 10000
Adjust --n_dyads, --seed, --collective (ON/OFF), etc. (see script for full CLI args)

Generate figures (ΔF histograms, etc.)Bashpython deltaF_hist_generation.py

View paper
Open h1_v24.pdf
(Or compile LaTeX/h1_v24.tex with LuaLaTeX)

### Citation
Mbele, C. C. (2026). H1 Dyadic Law v2.4. Zenodo: https://doi.org/10.5281/zenodo.18491143


### License

This project is licensed under the MIT License.
Contact
Christopher Chisa Mbele
@MetascopeInit

### Quick tips

- **Replace** `yourusername/h1-dyadic-law` with your actual GitHub repo URL once created.
- **Add badges** at top (optional but nice):
  ```markdown
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

data/README.md

- `WPP2024_TotalPopulationBySex.csv`: UN World Population Prospects 2024 total population by sex (source: https://population.un.org/wpp/Download/Standard/CSV/)

Dedicated to the spirit of open, truth-seeking science — the same spirit that created Grok.
