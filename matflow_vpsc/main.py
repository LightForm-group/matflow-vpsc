'`matflow_vpsc.main.py`'

from pathlib import Path

import numpy as np

from damask_parse.utils import validate_orientations

from matflow_vpsc import (
    input_mapper,
    output_mapper,
    cli_format_mapper,
    register_output_file,
    func_mapper,
    software_versions,
)
from matflow_vpsc.utils import quat2euler


@input_mapper('VPSC7.IN', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_in(path, load_case):
    pass


@input_mapper('FILECRYS', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_filecrys(path, load_case):
    pass


@input_mapper('FILETEXT', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_filetext(path, orientations):
    # Convert orientations and get size
    orientations = validate_orientations(orientations)
    euler_angles = quat2euler(orientations['quaternions'], degrees=True,
                              P=orientations['P'])
    num_oris = euler_angles.shape[0]

    # TODO: Check unit cell alignment! What conventions does VPSC code use?

    # Add weight column
    euler_angles = np.hstack((euler_angles, np.ones((num_oris, 1))))

    path = Path(path)
    with path.open(mode='w') as f:
        f.write("blank\nblank\nblank\n")
        f.write(f"B   {num_oris}\n")

        np.savetxt(f, euler_angles, fmt='%.2f')


@input_mapper('FILEPROC', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_fileproc(path, load_case):
    pass
