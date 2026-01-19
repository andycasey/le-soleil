#!/bin/bash
# Setup script for parallel HDF5 and MPI support
# Source this script before running MPI applications: source setup_parallel_hdf5.sh

echo "Activate venv"
source .venv/bin/activate

echo "Loading MPI and parallel HDF5 modules..."

# cephtweaks
module load cephtweaks

# Load OpenMPI (choose the version you prefer)
module load openmpi/5.0.6

# Load parallel HDF5
module load hdf5/mpi-1.14.5

# Set environment variables for h5py compilation with MPI
export HDF5_MPI="ON"
export CC=mpicc
export CEPHTWEAKS_LAZYIO=1

echo "Modules loaded:"
module list

echo ""
echo "Environment variables set:"
echo "  HDF5_MPI=$HDF5_MPI"
echo "  CC=$CC"
echo "  CEPHTWEAKS_LAZYIO=$CEPHTWEAKS_LAZYIO"


# Check if h5py is installed with MPI support
echo ""
echo "Checking h5py MPI support..."
python -c "import h5py; print('h5py MPI support:', h5py.get_config().mpi)" 2>/dev/null || echo "h5py not installed or import failed"

