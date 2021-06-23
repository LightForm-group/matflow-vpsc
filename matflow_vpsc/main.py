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

# {phase_name}.tex
# {phase_name}.sx
# part_i.proc
# vpsc7.in

@input_mapper('vpsc7.in', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_in(path, loadcase):
    phases = {
        'ti-hex': {
            'fraction': 1.0,
            'grain_shape_control': 0,
            'fragmentation_control': 0,
            'critical_aspect_ratio': 25,
            'init_ellipsoid_ratios': [1.0, 1.0, 1.0],
            'init_ellipsoid_ori': [0.0, 0.0, 0.0],
        },
        # 'ti-cub': {
        #     'fraction': 0.1,
        #     'grain_shape_control': 0,
        #     'fragmentation_control': 0,
        #     'critical_aspect_ratio': 25,
        #     'init_ellipsoid_ratios': [1.0, 1.0, 1.0],
        #     'init_ellipsoid_ori': [0.0, 0.0, 0.0],
        # }
    }
    numerics = {
        'errs': 0.001,
        'errd': 0.001,
        'errm': 0.001,
        'errso': 0.001,

        'itmaxext': 100,
        'itmaxint': 25,
        'itmaxso': 25,

        'irsvar': 0,
        'jrsini': 2,
        'jrsfin': 10,
        'jrstep': 2,

        'ibcinv': 1,
    }
    control = {
        'irecover': 0,
        'isave': 0,
        'icubcomp': 0,
        'nwrite': 0,

        'ihardlaw': 0,
        'iratesens': 1,
        'interaction': 4,
        'iupdate_ori': 1,
        'iupdate_shape': 1,
        'iupdate_hard': 1,
        'nneigh': 0,
        'iflu': 0,
    }

    phase_fractions = [phase['fraction'] for phase in phases.values()]
    assert(sum(phase_fractions) == 1.0)

    path = Path(path)
    with path.open(mode='w') as f:
        f.write('1 \n')
        f.write(f'{len(phases)}\n')
        f.write(' '.join(str(x) for x in phase_fractions) + '\n')

        for name, phase in phases.items():

            f.write('# Info on phase `{name}`\n')
            f.write(f'{phase['grain_shape_control']} '
                    f'{phase['fragmentation_control']} '
                    f'{phase['critical_aspect_ratio']}\n')
            f.write(' '.join(str(x) for x in phase['init_ellipsoid_ratios']) + '\n')
            f.write(' '.join(str(x) for x in phase['init_ellipsoid_ori']) + '\n')

            f.write('blank\n')
            f.write('{name}.tex')
            f.write('blank\n')
            f.write('{name}.sx')
            f.write('blank\n')
            f.write('blank\n')

        f.write('# Convergence paramenters\n')
        f.write(f'{numerics['errs']} {numerics['errd']} {numerics['errm']} '
                f'{numerics['errso']}\n')
        f.write(f'{numerics['itmaxext']} {numerics['itmaxint']} '
                f'{numerics['itmaxso']}\n')
        f.write(f'{numerics['irsvar']} {numerics['jrsini']} '
                f'{numerics['jrsfin']} {numerics['jrstep']}\n')
        
        f.write('# IO paramenters\n')
        f.write(f'{numerics['irecover']}\n')
        f.write(f'{numerics['isave']}\n')
        f.write(f'{numerics['icubcomp']}\n')
        f.write(f'{numerics['nwrite']}\n')

        f.write('# Model paramenters\n')
        f.write(f'{numerics['ihardlaw']}\n')
        f.write(f'{numerics['iratesens']}\n')
        f.write(f'{numerics['interaction']}\n')
        f.write(f'{numerics['iupdate_ori']} {numerics['iupdate_shape']} '
                f'{numerics['iupdate_hard']}\n')
        f.write(f'{numerics['nneigh']}\n')
        f.write(f'{numerics['iflu']}\n')

        f.write('# Process paramenters\n')
        f.write(f'{len(loadcase)}\n')
        f.write('blank\n')
        for i, range(len(loadcase)):
            f.write(f'0\n')
            f.write(f'part_{i+1}.proc\n')


@input_mapper('FILECRYS', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_filecrys(path):
    phases = {
        'ti-hex': {
            'lattice': 'hexagonal',
            'c/a': 1.624,
            'slip_modes': [{
                'name': 'basal',
                'n': 20,
                'tau_0': 3.,
                'tau_1': 1.,
                'theta_0': 5000.,
                'theta_1': 25.,
                'hpfac': 0.,
                'gndfac': 0.,
                'h_latent': [4., 4., 2., 4.],
                'slip_planes': [
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                ],
                'slip_dirs': [
                    [ 2, -1, -1, 0],
                    [-1,  2, -1, 0],
                    [-1, -1,  2, 0],
                ],
            }, {
                'name': 'prismatic',
                'n': 20,
                'tau_0': 14.,
                'tau_1': 40.,
                'theta_0': 590.,
                'theta_1': 50.,
                'hpfac': 0.,
                'gndfac': 0.,
                'h_latent': [4., 4., 2., 4.],
                'slip_planes': [
                    [ 1,  0, -1, 0],
                    [ 0, -1,  1, 0],
                    [-1,  1,  0, 0],
                ],
                'slip_dirs': [
                    [-1,  2, -1, 0],
                    [ 2, -1, -1, 0],
                    [-1, -1,  2, 0],
                ],
            }, {
                'name': 'pyramidal<c+a>',
                'n': 20,
                'tau_0': 44.,
                'tau_1': 100.,
                'theta_0': 5000.,
                'theta_1': 0.,
                'hpfac': 0.,
                'gndfac': 0.,
                'h_latent': [4., 4., 2., 4.],
                'slip_planes': [
                    [ 1,  0, -1, 1],
                    [ 1,  0, -1, 1],
                    [ 0, -1,  1, 1],
                    [ 0, -1,  1, 1],
                    [-1,  1,  0, 1],
                    [-1,  1,  0, 1],
                    [-1,  0,  1, 1],
                    [-1,  0,  1, 1],
                    [ 0,  1, -1, 1],
                    [ 0,  1, -1, 1],
                    [ 1, -1,  0, 1],
                    [ 1, -1,  0, 1],
                ],
                'slip_dirs': [
                    [-1, -1,  2, 3],
                    [-2,  1,  1, 3],
                    [ 1,  1, -2, 3],
                    [-1,  2, -1, 3],
                    [ 2, -1, -1, 3],
                    [ 1, -2,  1, 3],
                    [ 2, -1, -1, 3],
                    [ 1,  1, -2, 3],
                    [-1, -1,  2, 3],
                    [ 1, -2,  1, 3],
                    [-2,  1,  1, 3],
                    [-1,  2, -1, 3],
                ],
            }],
            'twin_modes': [{
                'name': 'tensile',
                'n': 20,

                'twsh': 0.13,
                'isectw': 0,
                'thres_1': 0.81,
                'thres_2': 0.,

                'tau_0': 44.,
                'tau_1': 100.,
                'theta_0': 5000.,
                'theta_1': 0.,
                'hpfac': 0.,
                'gndfac': 0.,
                'h_latent': [4., 4., 2., 4.],
                'slip_planes': [
                    [ 1,  0, -1, 2],
                    [ 0,  1, -1, 2],
                    [-1,  1,  0, 2],
                    [-1,  0,  1, 2],
                    [ 0, -1,  1, 2],
                    [ 1, -1,  0, 2],
                ],
                'slip_dirs': [
                    [-1,  0,  1, 1],
                    [ 0, -1,  1, 1],
                    [ 1, -1,  0, 1],
                    [ 1,  0, -1, 1],
                    [ 0,  1, -1, 1],
                    [-1,  1,  0, 1],
                ],
            }],
        },
        # 'ti-cub': {
        #     'fraction': 0.1,
        #     'grain_shape_control': 0,
        #     'fragmentation_control': 0,
        #     'critical_aspect_ratio': 25,
        #     'init_ellipsoid_ratios': [1.0, 1.0, 1.0],
        #     'init_ellipsoid_ori': [0.0, 0.0, 0.0],
        # }
    }


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
        f.write('blank\nblank\nblank\n')
        f.write(f'B   {num_oris}\n')

        np.savetxt(f, euler_angles, fmt='%.2f')


@input_mapper('FILEPROC', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_fileproc(path, load_case):
    path = Path(path)
    
    for i, load_part in enumerate(load_case):

        vel_grad = load_part.get('vel_grad')
        stress = load_part.get('stress')
        # rot = load_case.get('rotation')  # maybe implement later
        total_time = load_case['total_time']
        num_increments = load_case['num_increments']
        # freq = load_case.get('dump_frequency', 1)  # maybe implement later

        if vel_grad is None and stress is None:
            msg = ('Specify either `vel_grad`, `stress` or both.')
            raise ValueError(msg)

        msg = ('To use mixed boundary conditions, `{}` must be '
               'passed as a masked array.')
        if stress is None:
            # Just vel-grad

            if isinstance(vel_grad, np.ma.core.MaskedArray):
                raise ValueError(msg.format('stress'))

            vel_grad = np.ma.masked_array(vel_grad, mask=np.zeros((3, 3)))
            stress = np.ma.masked_array(np.zeros((3, 3)), np.ones((3, 3)))

        elif vel_grad is None:
            # Just stress

            if isinstance(stress, np.ma.core.MaskedArray):
                raise ValueError(msg.format('vel-grad'))

            vel_grad = np.ma.masked_array(np.zeros((3, 3)), np.ones((3, 3)))
            stress = np.ma.masked_array(stress, mask=np.zeros((3, 3)))

        else:
            # Both vel-grad and stress

            if not isinstance(vel_grad, np.ma.core.MaskedArray):
                raise ValueError(msg.format('vel_grad'))
            if not isinstance(stress, np.ma.core.MaskedArray):
                raise ValueError(msg.format('stress'))

            if np.any(dg_arr.mask == stress.mask):
                msg = ('`vel_grad` must be component-wise exclusive with '
                       '`stress`')
                raise ValueError(msg)

        time_increment = total_time / num_increments
        
        path_part = path.parent / f'part{i+1}.proc'
        with path_part.open(mode='w') as f:

            f.write(f' {num_increments} 7 {time_increment} 298.\n')
            f.write('blank\n')
            vel_grad_mask = np.logical_not(vel_grad.mask).astype(int)
            f.write(format_tensor33(vel_grad_mask))
            f.write('blank\n')
            f.write(format_tensor33(vel_grad, fmt='.3e'))
            f.write('blank\n')
            stress_mask = np.logical_not(stress.mask).astype(int)
            f.write(format_tensor33(stress_mask, sym=True))
            f.write('blank\n')
            f.write(format_tensor33(stress, fmt='.3e', sym=True))
