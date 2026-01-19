# le-soleil

MPI-parallel tool to convert 1D spectra from multiple instruments to a standardized HDF5 format.

## Installation

```bash
# Clone and setup
git clone https://github.com/andycasey/le-soleil.git
cd le-soleil
source setup.sh
```

## Usage

```bash
# Process all instruments with 32 MPI ranks
soleil -p 32 --kpf --neid --harpsn --expres -O spectra.h5

# Process only KPF with custom directory
soleil -p 16 --kpf --kpf-dir /data/kpf -O kpf.h5

# Test with limited files
soleil -p 4 --kpf --neid -l 100 -O test.h5 -f
```

### Options

```
-p, --nprocs N        Number of MPI processes (default: 1)
-O, --output FILE     Output HDF5 file (required)
-f, --overwrite       Overwrite existing output file
-l, --limit N         Limit number of files per instrument (for testing)

Instrument flags (at least one required):
  --kpf               Include KPF L1 data
  --neid              Include NEID L2 data
  --harpsn            Include HARPS-N S1D data
  --expres            Include EXPRES L1 data

Input directory overrides:
  --kpf-dir DIR       KPF input directory
  --neid-dir DIR      NEID input directory
  --harpsn-dir DIR    HARPS-N input directory
  --expres-dir DIR    EXPRES input directory
```

### Environment Variables

Set default input directories via environment variables:

```bash
export SOLEIL_KPF_DIR=/path/to/kpf/L1
export SOLEIL_NEID_DIR=/path/to/neid/L2
export SOLEIL_HARPSN_DIR=/path/to/harpsn/S1D
export SOLEIL_EXPRES_DIR=/path/to/expres/L1
```

## Output Format

Each instrument's data is stored in its own HDF5 group:

```
spectra.h5
├── kpf/
│   ├── wavelength    (n_spectra, n_orders, n_traces, n_pixels)
│   ├── flux          (n_spectra, n_orders, n_traces, n_pixels)
│   ├── ivar          (n_spectra, n_orders, n_traces, n_pixels)
│   ├── filename      (n_spectra,)
│   └── meta/
│       ├── DATE-OBS  (n_spectra,)
│       ├── MJD-OBS   (n_spectra,)
│       └── ...
├── neid/
│   └── ...
└── ...
```

### Dimensions

- `n_spectra`: Number of observations, sorted by observation epoch (MJD)
- `n_orders`: Number of spectral orders (GREEN + RED stacked for KPF)
- `n_traces`: Number of extraction traces (3 for KPF, 1 for others)
- `n_pixels`: Number of pixels per order

### Ordering

- **Wavelength along pixels**: blue → red
- **Wavelength along orders**: blue → red
- **Spectra**: sorted by observation epoch (increasing MJD)

## Supported Instruments

| Instrument | Data Level | Chips | Traces | Notes |
|------------|------------|-------|--------|-------|
| KPF        | L1         | GREEN + RED | 3 | Chips stacked along n_orders |
| NEID       | L2         | 1     | 1      | Science fiber only |
| HARPS-N    | S1D        | 1     | 1      | Stitched 1D spectrum |
| EXPRES     | L1         | 1     | 1      | Order-by-order |
