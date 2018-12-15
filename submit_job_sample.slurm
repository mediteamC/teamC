#!/bin/bash
#SBATCH -J just_a_test       # Job name
#SBATCH -o slurm.out       # Name of stdout output file
#SBATCH -e slurm.err      # Name of stderr error file
#SBATCH -p skx-dev      # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=gz2283@columbia.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

module load python3
python3 Pipline_templete.py
