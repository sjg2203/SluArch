#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Copyright (c) 2024. Simon J. Guillot. All rights reserved.                            +
#   Redistribution and use in source and binary forms, with or without modification, are strictly prohibited.
#                                                                                         +
#   THIS CODE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
#   BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#   OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
#   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS CODE,
#   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
SlµArch (Sleep microArchitecture): open-source package to analyse sleep microarchitecture.

- Author: Simon J. Guillot (https://sjg2203.github.io/)
- GitHub: https://github.com/sjg2203/sluarch
- License: Apache 2.0 License
"""

import logging

import mne
import numpy as np
import scipy.ndimage as ndimage
from mne.filter import filter_data
from scipy import signal
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.signal import spectrogram

LOGGING_TYPES = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)

logger = logging.getLogger('sluarch')

__all__ = ['spindles_abs', 'spindles_rel', 'slowosc', 'kcomp']


#############################################################################
# SLEEP SPINDLES
#############################################################################


def spindles_abs(raw, sf, thresh={'abs_pow': 1.25}) -> int:
    """Detects sleep spindles using absolute Sigma power.

    Notes
    ----------
    If you use this toolbox, please cite as followed:

    * Guillot S.J., (2024). SlµArch (2024.05.15). GitHub, Zenodo.
    https://doi.org/10.5281/zenodo.10066031

    We provide below some key points on the toolbox.

    If you have any questions, feel free to reach out on `GitHub <https://github.com/sjg2203/>`_.

    Example
    ----------
    >>> from sluarch import spindles_abs
    >>> #Load an EDF file using MNE
    >>> raw=mne.io.read_raw_edf("myfile.edf",preload=True)
    >>> sfreq=raw.info['sfreq']
    >>> #Return sleep spindles count into an array
    >>> spindles_abs(raw,sf=sfreq,thresh={'abs_pow':1.25})

    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    sf : float
        Sampling frequency in Hz.
    thresh : dict, optional
        Detection thresholds:
        ``'abs_pow'``: Absolute Sigma power (=power ratio freq_sigma/freq_broad; 1.25).

        If None is given, ``'abs_pow'``:0.2 automatically.

    Returns
    -------
    spindles_abs_count : int
        Count of sleep spindles using absolute Sigma power.
    """

    if 'abs_pow' not in thresh.keys():
        thresh['abs_pow'] = 1.25
    # If sf>100Hz, then data will be resampled to meet the 100Hz requirement
    if sf > 100:
        raw.resample(100)
        sf = 100
    else:
        sf = sf
    freq_broad = [1, 30]
    raw = raw.get_data()[0] * 1000000
    data = filter_data(
        raw, sf, freq_broad[0], freq_broad[1], method='fir', verbose=0
    )  # Apply a low and high bandpass filter
    data_sigma = data.copy()
    N = 20  # N order for the filter
    nyquist = sf / 2
    Wn = 11 / nyquist
    sos = signal.iirfilter(N, Wn, btype='Highpass', output='sos')
    data_sigma = signal.sosfiltfilt(sos, data_sigma)
    Wn = 16 / nyquist
    sos = signal.iirfilter(N, Wn, btype='lowpass', output='sos')
    data_sigma = signal.sosfiltfilt(sos, data_sigma)
    duration = 0.3
    halfduration = duration / 2
    total_duration = len(data_sigma) / sf
    last = len(data_sigma) - 1
    step = 0.1
    len_out = int(len(data_sigma) / (step * sf))
    out = np.zeros(len_out)
    tt = np.zeros(len_out)
    for i, j in enumerate(np.arange(0, total_duration, step)):
        beg = max(0, int((j - halfduration) * sf))
        end = min(last, int((j + halfduration) * sf))
        tt[i] = np.column_stack((beg, end)).mean(1) / sf
        out[i] = np.mean(np.square(data_sigma[beg:end]))
    dat_det_w = out
    dat_det_w[dat_det_w <= 0] = 0.000000001
    abs_sig_pow = np.log10(dat_det_w)
    interop = interp1d(tt, abs_sig_pow, kind='cubic', bounds_error=False, fill_value=0, assume_sorted=True)
    tt = np.arange(data_sigma.size) / sf
    abs_sig_pow = interop(tt)
    # Count of sleep spindles using absolute sigma power
    text = 'spindles'
    spindles_count_abs_pow = {}
    name = 0
    for item in abs_sig_pow:
        if item >= thresh['abs_pow']:
            spindles_count_abs_pow['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_count_abs_pow) == 1:
        text = 'spindle'
    spindles_abs_count = len(spindles_count_abs_pow)
    print('Using absolute Sigma power:', spindles_abs_count, text)
    return spindles_abs_count


def spindles_rel(raw, sf, thresh={'rel_pow': 0.2}) -> int:
    """Detects sleep spindles using relative Sigma power.

    Notes
    ----------
    If you use this toolbox, please cite as followed:

    * Guillot S.J., (2024). SlµArch (2024.05.15). GitHub, Zenodo.
    https://doi.org/10.5281/zenodo.10066031

    We provide below some key points on the toolbox.

    If you have any questions, feel free to reach out on `GitHub <https://github.com/sjg2203/>`_.

    Example
    ----------
    >>> from sluarch import spindles_rel
    >>> #Load an EDF file using MNE
    >>> raw=mne.io.read_raw_edf("myfile.edf",preload=True)
    >>> sfreq=raw.info['sfreq']
    >>> #Return sleep spindles count into an array
    >>> spindles_rel(raw,sf=sfreq,thresh={'rel_pow':0.2})

    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    sf : float
        Sampling frequency in Hz.
    thresh : dict, optional
        Detection thresholds:
        ``'rel_pow'``: Relative Sigma power (=power ratio freq_sigma/freq_broad; 0.2).

        If None is given, ``'rel_pow'``:0.2 automatically.

    Returns
    -------
    spindles_rel_count : int
        Count of sleep spindles using relative Sigma power.
    """

    if 'rel_pow' not in thresh.keys():
        thresh['rel_pow'] = 0.2
    # If sf>100Hz, then data will be resampled to meet the 100Hz requirement
    if sf > 100:
        raw.resample(100)
        sf = 100
    else:
        sf = sf
    freq_broad = [1, 30]
    raw = raw.get_data()[0] * 1000000
    data = filter_data(
        raw, sf, freq_broad[0], freq_broad[1], method='fir', verbose=0
    )  # Apply a low and high bandpass filter
    f, t, SXX = signal.stft(
        data, sf, nperseg=int((2 * sf)), noverlap=int(((2 * sf) - (0.2 * sf)))
    )  # Using STFT to compute the point-wise relative power
    idx_band = np.logical_and(
        f >= freq_broad[0], f <= freq_broad[1]
    )  # Keeping only the frequency of interest and Interpolating
    f = f[idx_band]
    SXX = SXX[idx_band, :]
    SXX = np.square(np.abs(SXX))
    sum_pow = SXX.sum(0).reshape(1, -1)
    np.divide(SXX, sum_pow, out=SXX)
    idx_sigma = np.logical_and(f >= 11, f <= 16)  # Extracting the relative sigma power
    rel_power = SXX[idx_sigma].sum(0)
    # Count of sleep spindles using relative sigma power
    text = 'spindles'
    spindles_count_rel_pow = {}
    name = 0
    for item in rel_power:
        if item >= thresh['rel_pow']:
            spindles_count_rel_pow['item' + str(name)] = [item]
        else:
            name += 1
    if len(spindles_count_rel_pow) == 1:
        text = 'spindle'
    spindles_rel_count = len(spindles_count_rel_pow)
    print('Using relative Sigma power:', spindles_rel_count, text)
    return spindles_rel_count


#############################################################################
# SOW OSCILLATIONS
#############################################################################


def slowosc(raw, sf) -> int:
    """Detects slow oscillations.

    Notes
    ----------
    If you use this toolbox, please cite as followed:

    * Guillot S.J., (2024). SlµArch (2024.05.15). GitHub, Zenodo.
    https://doi.org/10.5281/zenodo.10066031

    We provide below some key points on the toolbox.

    If you have any questions, feel free to reach out on `GitHub <https://github.com/sjg2203/>`_.

    Example
    ----------
    >>> from sluarch import slowosc
    >>> #Load an EDF file using MNE
    >>> raw=mne.io.read_raw_edf("myfile.edf",preload=True)
    >>> sfreq=raw.info['sfreq']
    >>> #Return sleep spindles count into an array
    >>> slowosc(raw,sf=sfreq)

    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    sf : float
        Sampling frequency in Hz.

    Returns
    -------
    int

        * ``'auc_sig'``: Sigma band AUC
        * ``'auc_psd'``: PSD AUC
        * ``'sos'``: Slow osclillations
    """

    # If sf>100Hz, then data will be resampled to meet the 100Hz requirement
    if sf > 100:
        raw.resample(100)
        sf = 100
    else:
        sf = sf
    freq_broad = [1, 45]
    raw = raw.get_data()[0] * 1000000
    data_filt = filter_data(
        raw, sf, freq_broad[0], freq_broad[1], method='fir', verbose=0
    )  # Apply a low and high bandpass filter
    data_sqz = data_filt.squeeze()
    win = 4 * sf
    low_sig, high_sig = 0.1, 2
    low_eeg, high_eeg = 0.1, 100
    freq, psd = signal.welch(data_sqz, sf, nperseg=win)
    idx_sigma = np.logical_and(freq >= low_sig, freq <= high_sig)
    idx_eeg = np.logical_and(freq >= low_eeg, freq <= high_eeg)
    freq_res = freq[1] - freq[0]  # 1/4=0.25
    auc_sig = simpson(psd[idx_sigma], dx=freq_res)
    auc_psd = simpson(psd[idx_eeg], dx=freq_res)
    sos = auc_sig / auc_psd
    return auc_sig, auc_psd, sos


#############################################################################
# K-COMPLEX
#############################################################################


def kcomp(raw, sf, Kplot=False) -> int:
    """Detects K-complex.

    Notes
    ----------
    If you use this toolbox, please cite as followed:

    * Guillot S.J., (2024). SlµArch (2024.05.15). GitHub, Zenodo.
    https://doi.org/10.5281/zenodo.10066031

    We provide below some key points on the toolbox.

    If you have any questions, feel free to reach out on `GitHub <https://github.com/sjg2203/>`_.

    Example
    ----------
    >>> from sluarch import kcomp
    >>> #Load an EDF file using MNE
    >>> raw=mne.io.read_raw_edf("myfile.edf",preload=True)
    >>> sfreq=raw.info['sfreq']
    >>> #Return sleep spindles count into an array
    >>> kcomp(raw,sf=sfreq,Kplot=False)

    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    sf : float
        Sampling frequency in Hz.
    Kplot : boolean
        If True, MNE backend will plot all the K-complex detected.
        Default is False.

    Returns
    -------
    nbK : int
        Count of K-complex.
    """

    if sf > 100:
        raw.resample(100)
        sf = 100
    else:
        sf = sf
    freq_broad = [1, 45]
    raw = raw.get_data()[0] * 1000000
    data_filt = filter_data(
        raw, sf, freq_broad[0], freq_broad[1], method='fir', verbose=0
    )  # Apply a low and high bandpass filter
    data_sqz = data_filt.squeeze()
    f, t, Sxx = spectrogram(data_sqz, sf, mode="psd", scaling="density")
    fmin = 0.3
    fmax = 1
    freq_slice = (f >= fmin) & (f <= fmax)
    # f_K=f[freq_slice]
    Sxx_K = Sxx[:, freq_slice].mean(axis=(0, 1))
    K_threshold = Sxx_K.mean()
    K_spikes = Sxx_K >= K_threshold
    labl, nbK = ndimage.label(K_spikes)
    if Kplot == True:
        start_list, stop_list = ([] for _ in range(2))
        for label in range(1, nbK + 1):
            start = np.where(labl == label)[0][0]
            stop = np.where(labl == label)[0][-1]
            start_list.append(start)
            stop_list.append(stop)
        K_events = np.zeros((len(start_list), 3), dtype=int)
        K_events[:, 0] = start_list
        K_events[:, 2] = 1
        K_epochs = mne.Epochs(
            raw, K_events, event_id={'K_complex': 1}, baseline=None, tmin=0, tmax=1, preload=True, verbose=False
        )
        K_epochs.plot()
    return nbK
