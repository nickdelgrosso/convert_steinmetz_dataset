from pathlib import Path
from typing import Any
import numpy as np
from tqdm import tqdm
from xarray import DataArray, Dataset, Coordinates


def steinmetz_to_xarray(dd: dict[str, Any]) -> Dataset:
    assert list(dd['ccf_axes']) == ['ap', 'dv', 'lr']
    dset = Dataset(
        dict(
            # Stimulus Data
            contrast_left = DataArray(
                data=(np.concatenate(
                    (dd['contrast_left'], dd['contrast_left_passive']),
                ) * 100).astype(np.int8),
                dims=('trial',)
            ),
            contrast_right = DataArray(
                data=(np.concatenate(
                    (dd['contrast_right'], dd['contrast_right_passive']),
                ) * 100).astype(np.int8),
                dims=('trial',)
            ),
            gocue = DataArray(
                data=np.concatenate((dd['gocue'].squeeze(), [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            stim_onset = DataArray(
                data=np.repeat([dd['stim_onset']], repeats=dd['active_trials'].shape[0]),
                dims=('trial'),
            ),
            feedback_type = DataArray(
                data=np.concatenate((dd['feedback_type'].squeeze(), [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            feedback_time = DataArray(
                data=np.concatenate((dd['feedback_time'].squeeze(), [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            response_type = DataArray(
                data=np.concatenate((dd['response'].squeeze(), [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            response_time = DataArray(
                data=np.concatenate((dd['response_time'].squeeze(), [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            reaction_type = DataArray(
                data=np.concatenate((dd['reaction_time'][:, 1], [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            reaction_time = DataArray(
                data=np.concatenate((dd['reaction_time'][:, 0], [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            prev_reward = DataArray(
                data=np.concatenate((dd['prev_reward'].squeeze(), [np.nan] * dd['licks_passive'].shape[1])), 
                dims=('trial',),
            ),
            active_trials = DataArray(data=dd['active_trials'], dims=('trial',)),

            # Wheel data
            wheel = DataArray(
                data=np.concatenate(
                    (dd['wheel'].squeeze(), dd['wheel_passive'].squeeze()), 
                    axis=0,
                ).astype(np.int8),
                dims=('trial', 'time')
            ),

            # Licks data
            licks = DataArray(
                data=np.concatenate(
                    (dd['licks'].squeeze(), dd['licks_passive'].squeeze()),
                    axis=0,
                ).astype(np.int8),
                dims=('trial', 'time'),
            ),

            # Pupil data
            pupil_x = DataArray(
                data=np.concatenate(
                    (dd['pupil'][0, :, :], dd['pupil_passive'][0, :, :]),
                    axis=0,
                ),
                dims=('trial', 'time')
            ),
            pupil_y = DataArray(
                data=np.concatenate(
                    (dd['pupil'][1, :, :], dd['pupil_passive'][1, :, :]),
                    axis=0,
                ),
                dims=('trial', 'time')
            ),
            pupil_area = DataArray(
                data=np.concatenate(
                    (dd['pupil'][2, :, :], dd['pupil_passive'][2, :, :]),
                    axis=0,
                ),
                dims=('trial', 'time')
            ),

            # Face data
            face = DataArray(
                data=np.concatenate(
                    (dd['face'].squeeze(), dd['face_passive'].squeeze()),
                    axis=0,
                ),
                dims=('trial', 'time'),
            ),

            # Spike data
            spks = DataArray(
                data=np.concatenate(
                    (dd['spks'], dd['spks_passive']),
                    axis=1,
                ).astype(np.int8), 
                dims=('cell', 'trial', 'time')
            ),
            trough_to_peak = DataArray(data=dd['trough_to_peak'].astype(np.int8), dims=('cell',)),
            ccf_ap = DataArray(data=dd['ccf'][:, 0], dims=('cell',)),
            ccf_dv = DataArray(data=dd['ccf'][:, 1], dims=('cell',)),
            ccf_lr = DataArray(data=dd['ccf'][:, 2], dims=('cell',)),
            brain_area = DataArray(data=dd['brain_area'], dims=('cell',)),
            
        ),
        coords=Coordinates({
            'trial': np.arange(1, dd['active_trials'].shape[0] + 1),
            'time': (np.arange(1, dd['wheel'].shape[-1] + 1) * dd['bin_size']),
            'cell': np.arange(1, dd['spks'].shape[0] + 1),
        }),
        attrs={
            'bin_size': dd['bin_size'],
            'stim_onset': dd['stim_onset'],
        }
    ).expand_dims({
        'mouse': [dd['mouse_name']],
        'session_date': [dd['date_exp']],
    })
    return dset



if __name__ == '__main__':

    base_path = Path('data/processed/neuropixels')
    base_path.mkdir(parents=True, exist_ok=True)

    for path in tqdm(list(Path('data/raw').glob('*.npz')), desc="Reading Raw NPZ Files"):
        dat = np.load(path, allow_pickle=True)['dat']

        for dd in tqdm(dat, desc=f"Writing Processed NetCDF Files from {path.name}"):
            dset = steinmetz_to_xarray(dd=dd)

            # Compression settings for each variable. 
            # Slower to write, but shrunk data to 6% the original size!
            settings = {'zlib': True, 'complevel': 5}
            encodings = {var: settings for var in dset.data_vars if not 'U' in str(dset[var].dtype)}
            
            dset.to_netcdf(
                path=base_path / f'steinmetz_{dd["date_exp"]}_{dd["mouse_name"]}.nc',
                format="NETCDF4",
                engine="netcdf4",
                encoding=encodings,   
            )
