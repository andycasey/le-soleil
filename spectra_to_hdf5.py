#!/usr/bin/env python3
"""
MPI-parallel script to load 1D spectra from various instruments and write to HDF5.

Supported instruments:
- KPF L1: Multi-extension FITS with GREEN/RED chips (stacked)
- NEID L2: Multi-extension FITS with SCIFLUX/WAVE
- HARPS-N S1D: FITS table with wavelength/flux columns
- EXPRES L1: FITS record array with nested spectra

Output format:
- Each instrument stored in its own HDF5 group (e.g., /kpf, /neid)
- Standardized dataset names: "wavelength", "flux", "ivar"
- Dimensions: (n_spectra, n_orders, n_traces, n_pixels)
- Wavelength ordering: blue -> red along both n_pixels and n_orders axes
- Spectra ordering: increasing observation epoch along n_spectra axis

Usage (via soleil CLI):
    soleil -p 32 --kpf --neid -O spectra.h5
"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from glob import glob
from typing import List, Dict, Tuple, Any
import numpy as np
import h5py
from mpi4py import MPI
from astropy.io import fits


# Default input directories (can be overridden via CLI or environment)
DEFAULT_DIRS = {
    "kpf": os.environ.get("SOLEIL_KPF_DIR", "/mnt/home/rrubenzahl/ceph/data/kpf/L1"),
    "neid": os.environ.get("SOLEIL_NEID_DIR", "/mnt/home/rrubenzahl/ceph/data/neid/L2"),
    "harpsn": os.environ.get("SOLEIL_HARPSN_DIR", "/mnt/home/rrubenzahl/ceph/data/harpsn/S1D"),
    "expres": os.environ.get("SOLEIL_EXPRES_DIR", "/mnt/home/rrubenzahl/ceph/data/expres/L1"),
}

# Common header keywords to extract (instrument-specific ones can be added)
COMMON_HEADER_KEYWORDS = [
    "DATE-OBS",
    "MJD-OBS",
    "OBJECT",
    "EXPTIME",
    "RA",
    "DEC",
]


class InstrumentHandler(ABC):
    """Base class for instrument-specific data handling.

    All handlers must return data in standardized format:
    - wavelength: (n_orders, n_traces, n_pixels) - blue to red ordering
    - flux: (n_orders, n_traces, n_pixels)
    - ivar: (n_orders, n_traces, n_pixels) - inverse variance
    """

    name: str = "base"

    @abstractmethod
    def discover_files(self, input_dir: str, limit: int = None) -> List[str]:
        """Discover all FITS files for this instrument"""
        pass

    @abstractmethod
    def get_data_shape(self, sample_file: str) -> Tuple[int, int, int]:
        """Get (n_orders, n_traces, n_pixels) from a sample file"""
        pass

    @abstractmethod
    def read_data(self, fits_file: str) -> Dict[str, np.ndarray]:
        """Read spectral data and return standardized arrays.

        Returns dict with keys: wavelength, flux, ivar
        Each array has shape (n_orders, n_traces, n_pixels)
        Arrays are ordered blue -> red along n_pixels and n_orders axes.
        """
        pass

    @abstractmethod
    def get_header_keywords(self) -> List[str]:
        """Return list of header keywords to extract"""
        pass

    def get_sort_key(self, fits_file: str) -> float:
        """Return a sort key for ordering by observation epoch (MJD)"""
        with fits.open(fits_file) as hdul:
            hdr = hdul[0].header
            mjd = hdr.get("MJD-OBS", None)
            if mjd is None:
                # Try DATE-OBS as fallback
                date_obs = hdr.get("DATE-OBS", "")
                if date_obs:
                    from astropy.time import Time
                    try:
                        mjd = Time(date_obs).mjd
                    except:
                        mjd = 0.0
                else:
                    mjd = 0.0
            return float(mjd)

    def read_header(self, fits_file: str) -> Dict[str, Any]:
        """Read header metadata from a FITS file"""
        metadata = {}
        with fits.open(fits_file) as hdul:
            hdr = hdul[0].header
            for key in self.get_header_keywords():
                try:
                    val = hdr.get(key, "")
                    metadata[key] = val if val is not None else ""
                except:
                    metadata[key] = ""
        return metadata


class KPFHandler(InstrumentHandler):
    """Handler for KPF L1 data.

    KPF has GREEN and RED chips, each with 3 extraction traces.
    Data is stacked: GREEN orders first, then RED orders (blue -> red).
    Output shape: (n_orders_total, n_traces=3, n_pixels)
    """

    name = "kpf"
    n_traces = 3

    def discover_files(self, input_dir: str, limit: int = None) -> List[str]:
        pattern = os.path.join(input_dir, "*", "*.fits")
        files = sorted(glob(pattern))
        return files[:limit] if limit else files

    def get_data_shape(self, sample_file: str) -> Tuple[int, int, int]:
        """Get (n_orders, n_traces, n_pixels) from a sample file."""
        with fits.open(sample_file) as hdul:
            # Get shape from GREEN chip (trace 1)
            green_data = hdul["GREEN_SCI_FLUX1"].data
            n_green_orders, n_pixels = green_data.shape
            # Get shape from RED chip
            red_data = hdul["RED_SCI_FLUX1"].data
            n_red_orders = red_data.shape[0]
            n_orders = n_green_orders + n_red_orders
            return (n_orders, self.n_traces, n_pixels)

    def read_data(self, fits_file: str) -> Dict[str, np.ndarray]:
        """Read and stack GREEN + RED chips into standardized arrays."""
        with fits.open(fits_file) as hdul:
            # Read all traces for both chips
            green_flux = []
            green_var = []
            green_wave = []
            red_flux = []
            red_var = []
            red_wave = []

            for i in range(1, self.n_traces + 1):
                green_flux.append(hdul[f"GREEN_SCI_FLUX{i}"].data.astype(np.float64))
                green_var.append(hdul[f"GREEN_SCI_VAR{i}"].data.astype(np.float64))
                green_wave.append(hdul[f"GREEN_SCI_WAVE{i}"].data.astype(np.float64))
                red_flux.append(hdul[f"RED_SCI_FLUX{i}"].data.astype(np.float64))
                red_var.append(hdul[f"RED_SCI_VAR{i}"].data.astype(np.float64))
                red_wave.append(hdul[f"RED_SCI_WAVE{i}"].data.astype(np.float64))

            # Stack traces: each is (n_orders, n_pixels) -> (n_orders, n_traces, n_pixels)
            green_flux = np.stack(green_flux, axis=1)
            green_var = np.stack(green_var, axis=1)
            green_wave = np.stack(green_wave, axis=1)
            red_flux = np.stack(red_flux, axis=1)
            red_var = np.stack(red_var, axis=1)
            red_wave = np.stack(red_wave, axis=1)

            # Stack GREEN + RED along orders axis (GREEN is bluer)
            flux = np.concatenate([green_flux, red_flux], axis=0)
            var = np.concatenate([green_var, red_var], axis=0)
            wavelength = np.concatenate([green_wave, red_wave], axis=0)

            # Ensure blue -> red ordering along pixels (check first order, first trace)
            if wavelength[0, 0, 0] > wavelength[0, 0, -1]:
                wavelength = wavelength[:, :, ::-1]
                flux = flux[:, :, ::-1]
                var = var[:, :, ::-1]

            # Ensure blue -> red ordering along orders (check central pixel)
            mid_pix = wavelength.shape[2] // 2
            if wavelength[0, 0, mid_pix] > wavelength[-1, 0, mid_pix]:
                wavelength = wavelength[::-1, :, :]
                flux = flux[::-1, :, :]
                var = var[::-1, :, :]

            # Convert variance to inverse variance
            with np.errstate(divide='ignore', invalid='ignore'):
                ivar = 1.0 / var
                ivar[~np.isfinite(ivar)] = 0.0

            return {"wavelength": wavelength, "flux": flux, "ivar": ivar}

    def get_header_keywords(self) -> List[str]:
        return COMMON_HEADER_KEYWORDS + ["TARGNAME", "GAIAID", "IMTYPE", "PROGNAME"]


class NEIDHandler(InstrumentHandler):
    """Handler for NEID L2 data.

    NEID has a single chip with science flux/variance/wavelength.
    Output shape: (n_orders, n_traces=1, n_pixels)
    """

    name = "neid"
    n_traces = 1

    def discover_files(self, input_dir: str, limit: int = None) -> List[str]:
        pattern = os.path.join(input_dir, "*", "*.fits")
        files = sorted(glob(pattern))
        return files[:limit] if limit else files

    def get_data_shape(self, sample_file: str) -> Tuple[int, int, int]:
        """Get (n_orders, n_traces, n_pixels) from a sample file."""
        with fits.open(sample_file) as hdul:
            data = hdul["SCIFLUX"].data
            n_orders, n_pixels = data.shape
            return (n_orders, self.n_traces, n_pixels)

    def read_data(self, fits_file: str) -> Dict[str, np.ndarray]:
        """Read NEID data into standardized arrays."""
        with fits.open(fits_file) as hdul:
            flux = hdul["SCIFLUX"].data.astype(np.float64)
            var = hdul["SCIVAR"].data.astype(np.float64)
            wavelength = hdul["SCIWAVE"].data.astype(np.float64)

            # Add trace dimension: (n_orders, n_pixels) -> (n_orders, 1, n_pixels)
            flux = flux[:, np.newaxis, :]
            var = var[:, np.newaxis, :]
            wavelength = wavelength[:, np.newaxis, :]

            # Ensure blue -> red ordering along pixels
            if wavelength[0, 0, 0] > wavelength[0, 0, -1]:
                wavelength = wavelength[:, :, ::-1]
                flux = flux[:, :, ::-1]
                var = var[:, :, ::-1]

            # Ensure blue -> red ordering along orders
            mid_pix = wavelength.shape[2] // 2
            if wavelength[0, 0, mid_pix] > wavelength[-1, 0, mid_pix]:
                wavelength = wavelength[::-1, :, :]
                flux = flux[::-1, :, :]
                var = var[::-1, :, :]

            # Convert variance to inverse variance
            with np.errstate(divide='ignore', invalid='ignore'):
                ivar = 1.0 / var
                ivar[~np.isfinite(ivar)] = 0.0

            return {"wavelength": wavelength, "flux": flux, "ivar": ivar}

    def get_header_keywords(self) -> List[str]:
        return COMMON_HEADER_KEYWORDS + ["QPROG", "DESSION", "OBSTYPE"]


class HARPSNHandler(InstrumentHandler):
    """Handler for HARPS-N S1D data (table format).

    HARPS-N S1D files contain 1D stitched spectra (not order-by-order).
    Output shape: (n_orders=1, n_traces=1, n_pixels)
    """

    name = "harpsn"
    n_orders = 1
    n_traces = 1

    def discover_files(self, input_dir: str, limit: int = None) -> List[str]:
        pattern = os.path.join(input_dir, "*", "*S1D*.fits")
        files = sorted(glob(pattern))
        return files[:limit] if limit else files

    def get_data_shape(self, sample_file: str) -> Tuple[int, int, int]:
        """Get (n_orders, n_traces, n_pixels) from a sample file."""
        with fits.open(sample_file) as hdul:
            table = hdul[1].data
            n_pixels = len(table)
            return (self.n_orders, self.n_traces, n_pixels)

    def read_data(self, fits_file: str) -> Dict[str, np.ndarray]:
        """Read HARPS-N data into standardized arrays."""
        with fits.open(fits_file) as hdul:
            table = hdul[1].data
            wavelength = table["wavelength"].astype(np.float64)
            flux = table["flux"].astype(np.float64)
            error = table["error"].astype(np.float64)

            # Reshape to (1, 1, n_pixels) for consistency
            wavelength = wavelength[np.newaxis, np.newaxis, :]
            flux = flux[np.newaxis, np.newaxis, :]
            error = error[np.newaxis, np.newaxis, :]

            # Ensure blue -> red ordering along pixels
            if wavelength[0, 0, 0] > wavelength[0, 0, -1]:
                wavelength = wavelength[:, :, ::-1]
                flux = flux[:, :, ::-1]
                error = error[:, :, ::-1]

            # Convert error to inverse variance
            with np.errstate(divide='ignore', invalid='ignore'):
                ivar = 1.0 / (error ** 2)
                ivar[~np.isfinite(ivar)] = 0.0

            return {"wavelength": wavelength, "flux": flux, "ivar": ivar}

    def get_header_keywords(self) -> List[str]:
        return COMMON_HEADER_KEYWORDS + ["HIERARCH ESO OBS PROG ID", "HIERARCH ESO OBS NAME"]


class EXPRESHandler(InstrumentHandler):
    """Handler for EXPRES L1 data (record array with nested spectra).

    EXPRES has order-by-order spectra with a single extraction trace.
    Output shape: (n_orders, n_traces=1, n_pixels)
    """

    name = "expres"
    n_traces = 1

    def discover_files(self, input_dir: str, limit: int = None) -> List[str]:
        pattern = os.path.join(input_dir, "*", "*.fits")
        files = sorted(glob(pattern))
        return files[:limit] if limit else files

    def get_data_shape(self, sample_file: str) -> Tuple[int, int, int]:
        """Get (n_orders, n_traces, n_pixels) from a sample file."""
        with fits.open(sample_file) as hdul:
            rec = hdul[1].data
            n_orders = len(rec)
            n_pixels = rec["spectrum"][0].shape[0]
            return (n_orders, self.n_traces, n_pixels)

    def read_data(self, fits_file: str) -> Dict[str, np.ndarray]:
        """Read EXPRES data into standardized arrays."""
        with fits.open(fits_file) as hdul:
            rec = hdul[1].data
            n_orders = len(rec)

            # Stack nested arrays into 2D array
            flux = np.stack([rec["spectrum"][i] for i in range(n_orders)]).astype(np.float64)
            uncertainty = np.stack([rec["uncertainty"][i] for i in range(n_orders)]).astype(np.float64)
            wavelength = np.stack([rec["wavelength"][i] for i in range(n_orders)]).astype(np.float64)

            # Add trace dimension: (n_orders, n_pixels) -> (n_orders, 1, n_pixels)
            flux = flux[:, np.newaxis, :]
            uncertainty = uncertainty[:, np.newaxis, :]
            wavelength = wavelength[:, np.newaxis, :]

            # Ensure blue -> red ordering along pixels
            if wavelength[0, 0, 0] > wavelength[0, 0, -1]:
                wavelength = wavelength[:, :, ::-1]
                flux = flux[:, :, ::-1]
                uncertainty = uncertainty[:, :, ::-1]

            # Ensure blue -> red ordering along orders
            mid_pix = wavelength.shape[2] // 2
            if wavelength[0, 0, mid_pix] > wavelength[-1, 0, mid_pix]:
                wavelength = wavelength[::-1, :, :]
                flux = flux[::-1, :, :]
                uncertainty = uncertainty[::-1, :, :]

            # Convert uncertainty to inverse variance
            with np.errstate(divide='ignore', invalid='ignore'):
                ivar = 1.0 / (uncertainty ** 2)
                ivar[~np.isfinite(ivar)] = 0.0

            return {"wavelength": wavelength, "flux": flux, "ivar": ivar}

    def get_header_keywords(self) -> List[str]:
        return COMMON_HEADER_KEYWORDS + ["OBSTYPE", "PROPID"]


# Registry of handlers
HANDLERS = {
    "kpf": KPFHandler,
    "neid": NEIDHandler,
    "harpsn": HARPSNHandler,
    "expres": EXPRESHandler,
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert spectra FITS files to HDF5 (MPI parallel)"
    )
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        required=True,
        choices=list(HANDLERS.keys()),
        help="Instruments to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files per instrument (for testing)",
    )
    # Per-instrument directory overrides
    parser.add_argument("--kpf-dir", type=str, default=None, help="KPF input directory")
    parser.add_argument("--neid-dir", type=str, default=None, help="NEID input directory")
    parser.add_argument("--harpsn-dir", type=str, default=None, help="HARPS-N input directory")
    parser.add_argument("--expres-dir", type=str, default=None, help="EXPRES input directory")
    return parser.parse_args()


def get_input_dir(instrument: str, args) -> str:
    """Get input directory for an instrument, checking CLI args, env vars, then defaults."""
    cli_dirs = {
        "kpf": args.kpf_dir,
        "neid": args.neid_dir,
        "harpsn": args.harpsn_dir,
        "expres": args.expres_dir,
    }
    # CLI override takes precedence
    if cli_dirs.get(instrument):
        return cli_dirs[instrument]
    # Fall back to default (which checks env vars)
    return DEFAULT_DIRS[instrument]


def create_instrument_group(
    f: h5py.File,
    instrument: str,
    n_files: int,
    data_shape: Tuple[int, int, int],
    header_keywords: List[str],
):
    """Create HDF5 group for an instrument with pre-allocated datasets."""
    n_orders, n_traces, n_pixels = data_shape
    full_shape = (n_files, n_orders, n_traces, n_pixels)

    grp = f.create_group(instrument)

    # Create standardized spectral data datasets
    for name in ["wavelength", "flux", "ivar"]:
        grp.create_dataset(
            name,
            shape=full_shape,
            dtype=np.float64,
            compression=None,
        )
        print(f"  Created /{instrument}/{name}: {full_shape}")

    # Create datasets for metadata (fixed-length strings for parallel HDF5)
    grp.create_dataset("filename", shape=(n_files,), dtype="S128")
    for key in header_keywords:
        # Sanitize key name for HDF5
        safe_key = key.replace(" ", "_").replace("/", "_")
        grp.create_dataset(f"meta/{safe_key}", shape=(n_files,), dtype="S128")

    # Store attributes
    grp.attrs["n_spectra"] = n_files
    grp.attrs["n_orders"] = n_orders
    grp.attrs["n_traces"] = n_traces
    grp.attrs["n_pixels"] = n_pixels

    print(f"  Created metadata datasets for {len(header_keywords)} keywords")


def process_instrument_parallel(
    handler: InstrumentHandler,
    fits_files: List[str],
    output_file: str,
    group_name: str,
    comm,
):
    """Process FITS files for one instrument in parallel and write to HDF5 group."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_files = len(fits_files)
    header_keywords = handler.get_header_keywords()

    # Each rank processes files at indices: rank, rank+size, rank+2*size, ...
    my_indices = list(range(rank, n_files, size))
    my_files = [fits_files[i] for i in my_indices]

    if rank == 0:
        print(f"  Processing {n_files} files with {size} MPI ranks")
        sys.stdout.flush()

    # Open file for parallel writing
    f = h5py.File(output_file, "r+", driver="mpio", comm=comm)
    grp = f[group_name]

    # Progress tracking
    tty = None
    if rank == 0:
        try:
            tty = open("/dev/tty", "w")
        except:
            pass

    try:
        from tqdm import tqdm
        iterator = tqdm(
            zip(my_indices, my_files),
            total=len(my_files),
            desc=f"  {group_name}",
            disable=(rank != 0 or tty is None),
            file=tty,
        )
    except ImportError:
        iterator = zip(my_indices, my_files)

    for idx, fits_file in iterator:
        try:
            # Read all spectral data at once (standardized format)
            data = handler.read_data(fits_file)

            # Write standardized spectral data
            grp["wavelength"][idx] = data["wavelength"]
            grp["flux"][idx] = data["flux"]
            grp["ivar"][idx] = data["ivar"]

            # Write filename
            grp["filename"][idx] = os.path.basename(fits_file).encode("utf-8")

            # Write header metadata
            metadata = handler.read_header(fits_file)
            for key, val in metadata.items():
                safe_key = key.replace(" ", "_").replace("/", "_")
                grp[f"meta/{safe_key}"][idx] = str(val).encode("utf-8")

        except Exception as e:
            print(f"Rank {rank}: Error processing {fits_file}: {e}")
            raise

    if tty:
        tty.close()

    f.close()
    comm.Barrier()


def main():
    """Main function"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t0 = datetime.now()

    args = parse_args()

    if rank == 0:
        print(f"Soleil - Spectra to HDF5 Converter")
        print(f"===================================")
        print(f"MPI ranks: {size}")
        print(f"Instruments: {', '.join(args.instruments)}")
        print(f"Output file: {args.output}")
        print(f"\nOutput format:")
        print(f"  Groups: /{', /'.join(args.instruments)}")
        print(f"  Datasets: wavelength, flux, ivar")
        print(f"  Dimensions: (n_spectra, n_orders, n_traces, n_pixels)")
        print(f"  Ordering: blue -> red along pixels and orders")
        print(f"  Spectra sorted by observation epoch (MJD)")
        sys.stdout.flush()

    # Prepare data for each instrument (rank 0 only, then broadcast)
    instrument_data = {}

    for instrument in args.instruments:
        handler = HANDLERS[instrument]()
        input_dir = get_input_dir(instrument, args)

        if rank == 0:
            print(f"\n[{instrument.upper()}]")
            print(f"  Input directory: {input_dir}")
            print(f"  Discovering FITS files...")

            fits_files = handler.discover_files(input_dir, args.limit)
            n_files = len(fits_files)

            if n_files == 0:
                print(f"  WARNING: No FITS files found for {instrument}. Skipping.")
                instrument_data[instrument] = None
                continue

            print(f"  Found {n_files} FITS files")

            # Sort files by observation epoch (MJD)
            print(f"  Sorting by observation epoch...")
            sort_keys = []
            for f in fits_files:
                try:
                    sort_keys.append(handler.get_sort_key(f))
                except:
                    sort_keys.append(0.0)
            sorted_indices = np.argsort(sort_keys)
            fits_files = [fits_files[i] for i in sorted_indices]
            print(f"  MJD range: {min(sort_keys):.2f} to {max(sort_keys):.2f}")

            # Get data shape from first file
            print(f"  Analyzing FITS structure...")
            data_shape = handler.get_data_shape(fits_files[0])
            n_orders, n_traces, n_pixels = data_shape
            print(f"    n_orders: {n_orders}, n_traces: {n_traces}, n_pixels: {n_pixels}")

            instrument_data[instrument] = {
                "files": fits_files,
                "shape": data_shape,
                "keywords": handler.get_header_keywords(),
            }
            sys.stdout.flush()

    # Broadcast instrument data to all ranks
    instrument_data = comm.bcast(instrument_data if rank == 0 else None, root=0)

    # Filter out instruments with no files
    active_instruments = [i for i in args.instruments if instrument_data.get(i) is not None]

    if not active_instruments:
        if rank == 0:
            print("\nNo files found for any instrument. Exiting.")
        sys.exit(1)

    # Check/create output file (rank 0 only)
    if rank == 0:
        if os.path.exists(args.output):
            if args.overwrite:
                print(f"\nRemoving existing output file: {args.output}")
                os.remove(args.output)
            else:
                print(f"\nOutput file exists: {args.output}")
                print("Use --overwrite to replace it.")
                comm.Abort(1)

        print(f"\nCreating output file: {args.output}")
        with h5py.File(args.output, "w") as f:
            # Store global attributes
            f.attrs["created"] = datetime.now().isoformat()
            f.attrs["instruments"] = active_instruments

            # Create group for each instrument
            for instrument in active_instruments:
                data = instrument_data[instrument]
                handler = HANDLERS[instrument]()
                print(f"\n  Creating group: /{instrument}")
                create_instrument_group(
                    f,
                    instrument,
                    len(data["files"]),
                    data["shape"],
                    data["keywords"],
                )
        sys.stdout.flush()

    comm.Barrier()

    # Process each instrument
    for instrument in active_instruments:
        if rank == 0:
            print(f"\nProcessing {instrument.upper()}...")
            sys.stdout.flush()

        handler = HANDLERS[instrument]()
        data = instrument_data[instrument]
        process_instrument_parallel(
            handler,
            data["files"],
            args.output,
            instrument,
            comm,
        )

        if rank == 0:
            print(f"  Completed {len(data['files'])} spectra")

    if rank == 0:
        dt = datetime.now() - t0
        print(f"\nTotal time: {dt}")
        print("Done!")


if __name__ == "__main__":
    main()
