'`matflow_vpsc.main.py`'

from pathlib import Path
from copy import copy

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
from matflow_vpsc.utils import quat2euler, format_tensor33


@input_mapper('vpsc7.in', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_in(path, control, phases, load_case, numerics):
    numerics_defaults = {
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
    control_defaults = {
        'irecover': 0,
        'isave': 0,
        'icubcomp': 0,
        'nwrite': 0,

        'ihardlaw': 0,
        'iratesens': 1,
        'interaction': 1,
        'iupdate_ori': 1,
        'iupdate_shape': 1,
        'iupdate_hard': 1,
        'nneigh': 0,
        'iflu': 0,
    }
    phase_defaults = {
        'fraction': 1.0,
        'grain_shape_control': 0,
        'fragmentation_control': 0,
        'critical_aspect_ratio': 25,
        'init_ellipsoid_ratios': [1.0, 1.0, 1.0],
        'init_ellipsoid_ori': [0.0, 0.0, 0.0],
    }

    if numerics is not None:
        numerics_defaults.update(numerics)
    numerics = numerics_defaults
    if control is not None:
        control_defaults.update(control)
    control = control_defaults
    phases_defaults = {}
    for name, phase in phases.items():
        phase_d = copy(phase_defaults)
        phase_d.update(phase)
        phases_defaults[name] = phase_d
    phases = phases_defaults

    phase_fractions = [phase['fraction'] for phase in phases.values()]
    assert(sum(phase_fractions) == 1.0)

    path = Path(path)
    with path.open(mode='w') as f:
        f.write('1 \n')
        f.write(f'{len(phases)}\n')
        f.write(' '.join(str(x) for x in phase_fractions) + '\n')

        for name, phase in phases.items():
            f.write(f'# Info on phase `{name}`\n')
            f.write(f'{phase["grain_shape_control"]} '
                    f'{phase["fragmentation_control"]} '
                    f'{phase["critical_aspect_ratio"]}\n')
            f.write(' '.join(str(x) for x in phase['init_ellipsoid_ratios']) + '\n')
            f.write(' '.join(str(x) for x in phase['init_ellipsoid_ori']) + '\n')

            f.write('blank\n')
            f.write(f'{name}.tex\n')
            f.write('blank\n')
            f.write(f'{name}.sx\n')
            f.write('blank\n')
            f.write('blank\n')

        f.write('# Convergence paramenters\n')
        f.write(f'{numerics["errs"]} {numerics["errd"]} {numerics["errm"]} '
                f'{numerics["errso"]}\n')
        f.write(f'{numerics["itmaxext"]} {numerics["itmaxint"]} '
                f'{numerics["itmaxso"]}\n')
        f.write(f'{numerics["irsvar"]} {numerics["jrsini"]} '
                f'{numerics["jrsfin"]} {numerics["jrstep"]}\n')
        f.write(f'{numerics["ibcinv"]}\n')

        f.write('# IO paramenters\n')
        f.write(f'{control["irecover"]}\n')
        f.write(f'{control["isave"]}\n')
        f.write(f'{control["icubcomp"]}\n')
        f.write(f'{control["nwrite"]}\n')

        f.write('# Model paramenters\n')
        f.write(f'{control["ihardlaw"]}\n')
        f.write(f'{control["iratesens"]}\n')
        f.write(f'{control["interaction"]}\n')
        f.write(f'{control["iupdate_ori"]} {control["iupdate_shape"]} '
                f'{control["iupdate_hard"]}\n')
        f.write(f'{control["nneigh"]}\n')
        f.write(f'{control["iflu"]}\n')

        f.write('# Process paramenters\n')
        f.write(f'{len(load_case)}\n')
        f.write('blank\n')
        for i in range(len(load_case)):
            f.write('0\n')
            f.write(f'part_{i+1}.proc\n')


@input_mapper('phase_name.sx', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_filecrys(path, phases):
    path = Path(path)

    for name, phase in phases.items():
        if phase['lattice'].lower() == 'hexagonal':
            lattice_params = f'1. 1. {phase["c/a"]} 90. 90. 120.\n'

            c_pars = phase['elastic_stiffness']
            c_pars['Z'] = 0.0
            c_pars['C_66'] = (c_pars['C_11'] - c_pars['C_12']) / 2.
            stiffness = '{C_11} {C_12} {C_13} {Z} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{C_12} {C_11} {C_13} {Z} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{C_13} {C_13} {C_33} {Z} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{Z} {Z} {Z} {C_44} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{Z} {Z} {Z} {Z} {C_44} {Z}\n'.format(**c_pars)
            stiffness += '{Z} {Z} {Z} {Z} {Z} {C_66}\n'.format(**c_pars)

        elif phase['lattice'].lower() == 'cubic':
            lattice_params = '1. 1. 1. 90. 90. 90.\n'

            c_pars = phase['elastic_stiffness']
            c_pars['Z'] = 0.0
            stiffness = '{C_11} {C_12} {C_12} {Z} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{C_12} {C_11} {C_12} {Z} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{C_12} {C_12} {C_11} {Z} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{Z} {Z} {Z} {C_44} {Z} {Z}\n'.format(**c_pars)
            stiffness += '{Z} {Z} {Z} {Z} {C_44} {Z}\n'.format(**c_pars)
            stiffness += '{Z} {Z} {Z} {Z} {Z} {C_44}\n'.format(**c_pars)

        else:
            msg = ('`lattice` must be `cubic`, `hexagonal`.')
            raise ValueError(msg)

        slip_modes = phase.get('slip_modes', [])
        twin_modes = phase.get('twin_modes', [])
        num_sl = len(slip_modes)
        num_tw = len(twin_modes)
        num_modes = num_sl + num_tw

        path_phase = path.parent / f'{name}.sx'
        with path_phase.open(mode='w') as f:
            f.write(f'# Material: {name}\n')
            f.write(f'{phase["lattice"].upper()}\n')
            f.write(lattice_params)
            f.write('# Elastic stiffness\n')
            f.write(stiffness)
            f.write('# Thermal expansion coefficients (ignored)\n')
            f.write('0.0 0.0 0.0 0.0 0.0 0.0\n')
            f.write('# Slip and twinning modes\n')
            f.write(f'{num_modes}\n')
            f.write(f'{num_modes}\n')
            f.write(' '.join(str(i) for i in range(1, num_modes + 1)) + '\n')
            for i, mode in enumerate(slip_modes + twin_modes):
                tw = i >= num_sl
                num_sys = len(mode['planes'])
                assert len(mode['planes']) == len(mode['directions'])
                f.write(f'{mode["name"]}\n')
                f.write(f'{i+1} {num_sys} {mode["n"]}')
                if tw:
                    f.write(' 0\n')
                    f.write(f'{mode["twsh"]} {mode["isectw"]} '
                            f'{mode["thres_1"]} {mode["thres_2"]}\n')
                else:
                    f.write(' 1\n')
                    f.write('0. 0 0. 0.\n')
                f.write(f'{mode["tau_0"]} {mode["tau_1"]} '
                        f'{mode["theta_0"]} {mode["theta_1"]} '
                        f'{mode["hpfac"]} {mode["gndfac"]}\n')
                f.write(' '.join(str(x) for x in mode['h_latent']) + '\n')
                for pl, dr in zip(mode['planes'], mode['directions']):
                    f.write(' '.join(str(x) for x in pl + dr) + '\n')


@input_mapper('phase_name.tex', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_filetext(path, orientations, phases):
    # Only working for single phase ATM
    path = Path(path)
    for name, phase in phases.items():
        path_tex = path.parent / f'{name}.tex'
        break

    # Convert orientations and get size
    orientations = validate_orientations(orientations)
    euler_angles = quat2euler(orientations['quaternions'], degrees=True,
                              P=orientations['P'])
    num_oris = euler_angles.shape[0]

    # TODO: Check unit cell alignment! What conventions does VPSC code use?
    # For hex phase check and convert unit cell alignment if necessary
    # VPSC uses x // a
    # (from code: 'c' coincident with 'z' and 'a' in the plane 'xz')
    if phase['lattice'].lower() == 'hexagonal':
        if 'unit_cell_alignment' not in orientations:
            msg = 'Orientation `unit_cell_alignment` must be specified.'
            raise ValueError(msg)

        if orientations['unit_cell_alignment'].get('y') == 'b':
            # Convert from y//b to x//a:
            euler_angles[:, 2] -= 30.
            euler_angles[euler_angles[:, 2] < 0., 2] += 360.

        elif orientations['unit_cell_alignment'].get('x') != 'a':
            msg = (f'Cannot convert from the following specified unit cell '
                   f'alignment to DAMASK-compatible unit cell alignment '
                   f'(x//a): {orientations["unit_cell_alignment"]}')
            NotImplementedError(msg)

    # Add weight column
    euler_angles = np.hstack((euler_angles, np.ones((num_oris, 1))))

    with path_tex.open(mode='w') as f:
        f.write('blank\nblank\nblank\n')
        f.write(f'B   {num_oris}\n')

        np.savetxt(f, euler_angles, fmt='%.2f')


@input_mapper('part_i.proc', 'simulate_orientations_loading', 'self_consistent')
def write_vpsc_fileproc(path, load_case):
    path = Path(path)

    for i, load_part in enumerate(load_case):

        vel_grad = load_part.get('vel_grad')
        stress = load_part.get('stress')
        # rot = load_part.get('rotation')  # maybe implement later
        total_time = load_part['total_time']
        num_increments = load_part['num_increments']
        # freq = load_part.get('dump_frequency', 1)  # maybe implement later

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

            if np.any(vel_grad.mask == stress.mask):
                msg = ('`vel_grad` must be component-wise exclusive with '
                       '`stress`')
                raise ValueError(msg)

        time_increment = total_time / num_increments

        path_part = path.parent / f'part_{i+1}.proc'
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


@output_mapper(
    'orientations_response',
    'simulate_orientations_loading',
    'self_consistent'
)
def read_output_files(path):
    volume_response = {}

    # Read orientations output file
    # num_phases = 1
    # for phase in range(num_phases):
    phase = 0
    path_tex = path.parent / f'TEX_PH{phase+1}.OUT'

    num_oris = None
    all_strain = []
    all_elps_axes = []
    all_elps_ori = []
    all_oris = []
    with path_tex.open(mode='r') as f:
        try:
            while True:
                strain = float(next(f).split()[-1])
                # Skip last inc if it is repeated
                if num_oris is not None and strain == all_strain[-1]:
                    break
                elps_axes = [float(x) for x in next(f).split()[:3]]
                elps_ori = [float(x) for x in next(f).split()[:3]]
                conv, num_oris_in = next(f).split()
                num_oris_in = int(num_oris_in)

                assert conv == 'B'
                if num_oris is None:
                    num_oris = num_oris_in
                else:
                    assert num_oris_in == num_oris

                oris = np.loadtxt(f, max_rows=num_oris, usecols=[0, 1, 2])

                all_strain.append(strain)
                all_elps_axes.append(elps_axes)
                all_elps_ori.append(elps_ori)
                all_oris.append(oris)
        except StopIteration:
            pass

    all_strain = np.array(all_strain)
    all_elps_axes = np.array(all_elps_axes)
    all_elps_ori = np.array(all_elps_ori)
    all_oris = np.array(all_oris)

    volume_response.update({
        'O': {
            'data': {
                'type': 'euler',
                'euler_angles': all_oris,
                # TODO: check this!!
                'unit_cell_alignment': {'x': 'a'},
                'euler_degrees': True,
            },
            'meta': {
                'increments': list(range(1, all_oris.shape[0] + 1)),
            }
        }
    })

    # Read average stress/strain output file
    path_strstr = path.parent / 'STR_STR.OUT'
    strstr_data = np.loadtxt(path_strstr, skiprows=1)

    orientations_response = {
        'volume_data': volume_response,
    }
    return orientations_response
