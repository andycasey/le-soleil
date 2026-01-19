  # Source the environment
  source setup.sh

  # KPF (90k files, ~18 extensions per file)
  mpirun -np 32 python spectra_to_hdf5.py --instrument kpf --output /mnt/home/acasey/ceph/kpf_spectra.h5

  # NEID
  mpirun -np 32 python spectra_to_hdf5.py --instrument neid --output /mnt/home/acasey/ceph/neid_spectra.h5

  # HARPS-N
  mpirun -np 32 python spectra_to_hdf5.py --instrument harpsn --output
  /mnt/home/acasey/ceph/harpsn_spectra.h5

  # EXPRES
  mpirun -np 32 python spectra_to_hdf5.py --instrument expres --output
  /mnt/home/acasey/ceph/expres_spectra.h5

  # Test with limited files
  mpirun -np 4 python spectra_to_hdf5.py --instrument kpf --output test.h5 --limit 100 --overwrite
