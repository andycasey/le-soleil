#!/usr/bin/env python3
"""
Validate HDF5 output from soleil.

Checks:
- All expected instrument groups exist
- Each group has wavelength, flux, ivar datasets
- Dataset shapes match (n_spectra, n_orders, n_traces, n_pixels)
- Metadata datasets exist
- Wavelength increases blue -> red along pixels axis
- Wavelength increases blue -> red along orders axis
"""

import sys
import numpy as np
import h5py


def validate_wavelength_ordering(wavelength: np.ndarray, instrument: str) -> bool:
    """
    Validate that wavelength increases blue -> red.

    Args:
        wavelength: Array of shape (n_spectra, n_orders, n_traces, n_pixels)
        instrument: Instrument name for error messages

    Returns:
        True if validation passes
    """
    n_spectra, n_orders, n_traces, n_pixels = wavelength.shape

    # Check pixel ordering (blue -> red along last axis)
    # Sample a few spectra to check
    n_check = min(5, n_spectra)
    for i in range(n_check):
        for j in range(n_orders):
            for k in range(n_traces):
                wl = wavelength[i, j, k, :]
                # Skip if all NaN or zero
                valid = np.isfinite(wl) & (wl > 0)
                if np.sum(valid) < 2:
                    continue
                wl_valid = wl[valid]
                if wl_valid[0] >= wl_valid[-1]:
                    print(f"  FAIL: {instrument} wavelength not increasing along pixels")
                    print(f"        spectrum={i}, order={j}, trace={k}")
                    print(f"        first={wl_valid[0]:.2f}, last={wl_valid[-1]:.2f}")
                    return False

    print(f"  PASS: Wavelength increases blue -> red along pixels")

    # Check order ordering (blue -> red along orders axis)
    # Compare central wavelength of first and last order
    if n_orders > 1:
        mid_pix = n_pixels // 2
        for i in range(n_check):
            for k in range(n_traces):
                wl_first = wavelength[i, 0, k, mid_pix]
                wl_last = wavelength[i, -1, k, mid_pix]
                # Skip if invalid
                if not (np.isfinite(wl_first) and np.isfinite(wl_last)):
                    continue
                if wl_first <= 0 or wl_last <= 0:
                    continue
                if wl_first >= wl_last:
                    print(f"  FAIL: {instrument} wavelength not increasing along orders")
                    print(f"        spectrum={i}, trace={k}")
                    print(f"        first_order_mid={wl_first:.2f}, last_order_mid={wl_last:.2f}")
                    return False

        print(f"  PASS: Wavelength increases blue -> red along orders")
    else:
        print(f"  SKIP: Only 1 order, skipping order direction check")

    return True


def validate_instrument_group(f: h5py.File, instrument: str) -> bool:
    """
    Validate a single instrument group in the HDF5 file.

    Returns:
        True if all validations pass
    """
    print(f"\n[{instrument.upper()}]")

    if instrument not in f:
        print(f"  SKIP: Group /{instrument} not found (no data for this instrument)")
        return True

    grp = f[instrument]

    # Check required datasets exist
    required_datasets = ["wavelength", "flux", "ivar", "filename"]
    for ds in required_datasets:
        if ds not in grp:
            print(f"  FAIL: Missing dataset '{ds}'")
            return False
    print(f"  PASS: All required datasets present")

    # Check shapes match
    wl_shape = grp["wavelength"].shape
    flux_shape = grp["flux"].shape
    ivar_shape = grp["ivar"].shape

    if wl_shape != flux_shape or wl_shape != ivar_shape:
        print(f"  FAIL: Shape mismatch")
        print(f"        wavelength: {wl_shape}")
        print(f"        flux: {flux_shape}")
        print(f"        ivar: {ivar_shape}")
        return False

    n_spectra, n_orders, n_traces, n_pixels = wl_shape
    print(f"  PASS: Shapes match: ({n_spectra}, {n_orders}, {n_traces}, {n_pixels})")

    # Check dimensions are reasonable
    if n_spectra == 0:
        print(f"  FAIL: n_spectra is 0")
        return False
    if n_orders == 0:
        print(f"  FAIL: n_orders is 0")
        return False
    if n_traces == 0:
        print(f"  FAIL: n_traces is 0")
        return False
    if n_pixels == 0:
        print(f"  FAIL: n_pixels is 0")
        return False
    print(f"  PASS: All dimensions > 0")

    # Check filename dataset
    if grp["filename"].shape[0] != n_spectra:
        print(f"  FAIL: filename length ({grp['filename'].shape[0]}) != n_spectra ({n_spectra})")
        return False
    print(f"  PASS: filename array length matches n_spectra")

    # Check metadata group exists
    if "meta" not in grp:
        print(f"  FAIL: Missing 'meta' group")
        return False

    meta_keys = list(grp["meta"].keys())
    if len(meta_keys) == 0:
        print(f"  FAIL: No metadata datasets in 'meta' group")
        return False
    print(f"  PASS: Metadata group exists with {len(meta_keys)} keys")

    # Check attributes
    expected_attrs = ["n_spectra", "n_orders", "n_traces", "n_pixels"]
    for attr in expected_attrs:
        if attr not in grp.attrs:
            print(f"  FAIL: Missing attribute '{attr}'")
            return False
    print(f"  PASS: All expected attributes present")

    # Validate attribute values match actual shape
    if grp.attrs["n_spectra"] != n_spectra:
        print(f"  FAIL: n_spectra attribute ({grp.attrs['n_spectra']}) != actual ({n_spectra})")
        return False
    if grp.attrs["n_orders"] != n_orders:
        print(f"  FAIL: n_orders attribute ({grp.attrs['n_orders']}) != actual ({n_orders})")
        return False
    if grp.attrs["n_traces"] != n_traces:
        print(f"  FAIL: n_traces attribute ({grp.attrs['n_traces']}) != actual ({n_traces})")
        return False
    if grp.attrs["n_pixels"] != n_pixels:
        print(f"  FAIL: n_pixels attribute ({grp.attrs['n_pixels']}) != actual ({n_pixels})")
        return False
    print(f"  PASS: Attributes match actual dimensions")

    # Validate wavelength ordering
    wavelength = grp["wavelength"][:]
    if not validate_wavelength_ordering(wavelength, instrument):
        return False

    # Check that flux and ivar have some valid data
    flux = grp["flux"][:]
    ivar = grp["ivar"][:]

    valid_flux = np.isfinite(flux) & (flux != 0)
    valid_ivar = np.isfinite(ivar) & (ivar >= 0)

    flux_coverage = np.sum(valid_flux) / flux.size * 100
    ivar_coverage = np.sum(valid_ivar) / ivar.size * 100

    print(f"  INFO: Flux coverage: {flux_coverage:.1f}% valid")
    print(f"  INFO: Ivar coverage: {ivar_coverage:.1f}% valid (>=0)")

    if flux_coverage < 1:
        print(f"  WARN: Very low flux coverage (<1%)")
    if ivar_coverage < 1:
        print(f"  WARN: Very low ivar coverage (<1%)")

    return True


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <hdf5_file>")
        sys.exit(1)

    hdf5_file = sys.argv[1]

    print(f"Validating: {hdf5_file}")

    try:
        f = h5py.File(hdf5_file, "r")
    except Exception as e:
        print(f"FAIL: Cannot open HDF5 file: {e}")
        sys.exit(1)

    # Check global attributes
    print("\n[GLOBAL]")
    if "created" not in f.attrs:
        print("  WARN: Missing 'created' attribute")
    else:
        print(f"  PASS: Created timestamp: {f.attrs['created']}")

    if "instruments" not in f.attrs:
        print("  FAIL: Missing 'instruments' attribute")
        sys.exit(1)

    instruments = list(f.attrs["instruments"])
    print(f"  PASS: Instruments attribute: {instruments}")

    # Validate each instrument
    all_passed = True
    instruments_found = 0

    for instrument in ["kpf", "neid", "harpsn", "expres"]:
        if instrument in f:
            instruments_found += 1
            if not validate_instrument_group(f, instrument):
                all_passed = False

    f.close()

    print("\n" + "=" * 40)
    if instruments_found == 0:
        print("FAIL: No instrument groups found in file")
        sys.exit(1)
    elif all_passed:
        print(f"SUCCESS: All {instruments_found} instrument(s) validated")
        sys.exit(0)
    else:
        print("FAIL: Some validations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
