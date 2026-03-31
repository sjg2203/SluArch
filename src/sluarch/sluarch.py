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

- Author: Simon J. Guillot (https://sjg2203.github.io)
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

LOGGING_TYPES=dict(DEBUG=logging.DEBUG,INFO=logging.INFO,WARNING=logging.WARNING,ERROR=logging.ERROR,CRITICAL=logging.CRITICAL,)

logger=logging.getLogger('sluarch')

__all__=['spindles_abs','spindles_rel','spindles_rms','spindles_cov','slowosc','kcomp']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_raw(raw: mne.io.BaseRaw,sf: float) -> None:
	"""Validate common inputs shared by all detection functions."""
	if not isinstance(raw,mne.io.BaseRaw):
		raise TypeError(f'raw must be an MNE BaseRaw instance, got {type(raw).__name__!r}.')
	if sf<=0:
		raise ValueError(f'sf must be a positive sampling frequency, got {sf}.')


def _maybe_resample(raw: mne.io.BaseRaw,sf: float,target: int) -> tuple[mne.io.BaseRaw,float]:
	"""Return a resampled copy of *raw* only when ``sf`` exceeds *target* Hz."""
	if sf>target:
		raw=raw.copy().resample(target,verbose=False)
		sf=float(target)
	return raw,sf


def _label_events(mask: np.ndarray,sf: float,min_dur: float = 0.5,max_dur: float = 3.0,) -> int:
	"""Count contiguous ``True`` runs in *mask* that satisfy duration bounds.

	Parameters
	----------
	mask :
		Boolean array with one entry per sample.
	sf :
		Sampling frequency in Hz.
	min_dur :
		Minimum event duration in seconds (inclusive).
	max_dur :
		Maximum event duration in seconds (inclusive).

	Returns
	-------
	int
		Number of events whose duration falls within [min_dur, max_dur].
	"""
	labels,n_labels=ndimage.label(mask)
	count=0
	for lbl in range(1,n_labels+1):
		dur=np.sum(labels==lbl)/sf
		if min_dur<=dur<=max_dur:
			count+=1
	return count


def _sliding_window(x: np.ndarray,win: int,step: int) -> np.ndarray:
	"""Return a strided 2-D view of *x* shaped ``(n_windows, win)``.

	Uses :func:`numpy.lib.stride_tricks.sliding_window_view` (NumPy ≥ 1.20),
	which produces a zero-copy read-only view — no data is copied until the
	caller performs a reduction (e.g. ``mean``, ``std``).

	Parameters
	----------
	x :
		1-D input array.
	win :
		Window length in samples.
	step :
		Step between successive windows in samples.

	Returns
	-------
	numpy.ndarray
		Shape ``(n_windows, win)`` strided view.
	"""
	return np.lib.stride_tricks.sliding_window_view(x,win)[::step]


# ---------------------------------------------------------------------------
# SLEEP SPINDLES
# ---------------------------------------------------------------------------


def spindles_abs(raw: mne.io.BaseRaw,sf: float,thresh: dict|None = None,freq_sigma: tuple[
	float,float] = (10.0,15.0),min_dur: float = 0.5,max_dur: float = 3.0,resample_to: int = 512,) -> int:
	"""Detect sleep spindles using absolute Sigma power.

	A short-time RMS envelope is computed on the band-pass filtered signal,
	log10-transformed, and thresholded.  Contiguous supra-threshold epochs
	that satisfy the duration criteria are counted as individual spindle events.

	Notes
	----------
	If you use this toolbox, please cite as follows:

	* Guillot S.J., (2024). SlµArch. GitHub, Zenodo.
	  https://doi.org/10.5281/zenodo.11204061

	Example
	-------
	>>> from sluarch import spindles_abs
	>>> raw = mne.io.read_raw_edf('myfile.edf', preload=True)
	>>> spindles_abs(raw, sf=raw.info['sfreq'])

	Parameters
	----------
	raw :
		An MNE Raw instance.  The first EEG channel is used.
	sf :
		Sampling frequency in Hz.
	thresh :
		Detection threshold dict.  Key ``'abs_pow'`` sets the log10 absolute
		sigma-power threshold (default ``1.25``).  Pass ``None`` to use the
		default.
	freq_sigma :
		Sigma band ``(low, high)`` in Hz.  Default ``(10, 15)``Hz.
	min_dur :
		Minimum spindle duration in seconds (default ``0.5``s).
	max_dur :
		Maximum spindle duration in seconds (default ``3.0``s).
	resample_to :
		Target sampling rate in Hz applied when ``sf`` exceeds this value
		(default ``512``Hz).

	Returns
	-------
	int
		Number of detected sleep spindles.
	"""
	_validate_raw(raw,sf)
	if thresh is None:
		thresh={}
	thresh.setdefault('abs_pow',1.25)
	if not (0<freq_sigma[0]<freq_sigma[1]):
		raise ValueError('freq_sigma must satisfy 0 < low < high.')
	raw,sf=_maybe_resample(raw,sf,resample_to)
	freq_broad=(1.0,30.0)
	data=raw.get_data()[0]*1_000_000
	data=filter_data(data,sf,freq_broad[0],freq_broad[1],method='fir',verbose=False)
	nyquist=sf/2.0
	N=20
	sos_hi=signal.iirfilter(N,freq_sigma[0]/nyquist,btype='highpass',output='sos')
	data_sigma=signal.sosfiltfilt(sos_hi,data)
	sos_lo=signal.iirfilter(N,freq_sigma[1]/nyquist,btype='lowpass',output='sos')
	data_sigma=signal.sosfiltfilt(sos_lo,data_sigma)
	duration=0.3
	halfduration=duration/2.0
	total_duration=len(data_sigma)/sf
	last=len(data_sigma)-1
	step=0.1
	len_out=int(len(data_sigma)/(step*sf))
	out=np.zeros(len_out)
	tt=np.zeros(len_out)
	for i,j in enumerate(np.arange(0,total_duration,step)):
		beg=max(0,int((j-halfduration)*sf))
		end=min(last,int((j+halfduration)*sf))
		tt[i]=np.mean([beg,end])/sf
		out[i]=np.mean(np.square(data_sigma[beg:end]))
	out=np.where(out>0,out,1e-9)
	abs_sig_pow=np.log10(out)
	interop=interp1d(tt,abs_sig_pow,kind='cubic',bounds_error=False,fill_value=0,assume_sorted=True)
	tt_full=np.arange(data_sigma.size)/sf
	abs_sig_pow_full=interop(tt_full)
	mask=abs_sig_pow_full>=thresh['abs_pow']
	count=_label_events(mask,sf,min_dur=min_dur,max_dur=max_dur)
	text='spindle' if count==1 else 'spindles'
	logger.info('Using absolute Sigma power: %d %s',count,text)
	print(f'Using absolute Sigma power: {count} {text}')
	return count


def spindles_rel(raw: mne.io.BaseRaw,sf: float,thresh: dict|None = None,freq_sigma: tuple[
	float,float] = (10.0,15.0),min_dur: float = 0.5,max_dur: float = 3.0,resample_to: int = 512,) -> int:
	"""Detect sleep spindles using relative Sigma power.

	Relative power is computed via STFT and expressed as the ratio of sigma
	band power to broadband power.  Contiguous supra-threshold STFT frames
	that satisfy the duration criteria are counted as individual spindle events.

	Notes
	----------
	If you use this toolbox, please cite as follows:

	* Guillot S.J., (2024). SlµArch. GitHub, Zenodo.
	  https://doi.org/10.5281/zenodo.11204061

	Example
	-------
	>>> from sluarch import spindles_rel
	>>> raw = mne.io.read_raw_edf('myfile.edf', preload=True)
	>>> spindles_rel(raw, sf=raw.info['sfreq'])

	Parameters
	----------
	raw :
		An MNE Raw instance.  The first EEG channel is used.
	sf :
		Sampling frequency in Hz.
	thresh :
		Detection threshold dict.  Key ``'rel_pow'`` sets the relative sigma
		power threshold (default ``0.2``).  Pass ``None`` to use the default.
	freq_sigma :
		Sigma band ``(low, high)`` in Hz.  Default ``(10, 15)``Hz.
	min_dur :
		Minimum spindle duration in seconds (default ``0.5``s).
	max_dur :
		Maximum spindle duration in seconds (default ``3.0``s).
	resample_to :
		Target sampling rate in Hz applied when ``sf`` exceeds this value
		(default ``512``Hz).

	Returns
	-------
	int
		Number of detected sleep spindles.
	"""
	_validate_raw(raw,sf)
	if thresh is None:
		thresh={}
	thresh.setdefault('rel_pow',0.2)
	if not (0<freq_sigma[0]<freq_sigma[1]):
		raise ValueError('freq_sigma must satisfy 0 < low < high.')
	raw,sf=_maybe_resample(raw,sf,resample_to)
	freq_broad=(1.0,30.0)
	data=raw.get_data()[0]*1_000_000
	data=filter_data(data,sf,freq_broad[0],freq_broad[1],method='fir',verbose=False)
	nperseg=int(2*sf)
	noverlap=int(nperseg-0.2*sf)
	f,_t,SXX=signal.stft(data,sf,nperseg=nperseg,noverlap=noverlap)
	idx_broad=np.logical_and(f>=freq_broad[0],f<=freq_broad[1])
	f=f[idx_broad]
	SXX=np.square(np.abs(SXX[idx_broad,:]))
	sum_pow=SXX.sum(axis=0,keepdims=True)
	sum_pow=np.where(sum_pow>0,sum_pow,1e-9)
	SXX=np.divide(SXX,sum_pow)
	idx_sigma=np.logical_and(f>=freq_sigma[0],f<=freq_sigma[1])
	rel_power=SXX[idx_sigma].sum(axis=0)
	frame_dur=(nperseg-noverlap)/sf
	mask=rel_power>=thresh['rel_pow']
	labels,n_labels=ndimage.label(mask)
	count=0
	for lbl in range(1,n_labels+1):
		dur=np.sum(labels==lbl)*frame_dur
		if min_dur<=dur<=max_dur:
			count+=1
	text='spindle' if count==1 else 'spindles'
	logger.info('Using relative Sigma power: %d %s',count,text)
	print(f'Using relative Sigma power: {count} {text}')
	return count


def spindles_rms(raw: mne.io.BaseRaw,sf: float,n_sd: float = 5.0,freq_sigma: tuple[
	float,float] = (10.0,15.0),win_sec: float = 0.3,step_sec: float = 0.1,min_dur: float = 0.5,max_dur: float = 3.0,resample_to: int = 512,) -> int:
	"""Detect sleep spindles using a moving-RMS envelope with SD-based threshold.

	The sigma-filtered signal's RMS is computed in overlapping windows and the
	detection threshold is set adaptively as::

	    threshold = mean(RMS) + n_sd × std(RMS)

	This data-driven threshold scales automatically with the recording's
	amplitude so no fixed power value needs calibrating per rig.

	Notes
	-----
	If you use this toolbox, please cite as follows:

	* Guillot S.J., (2024). SlµArch. GitHub, Zenodo.
	  https://doi.org/10.5281/zenodo.11204061

	Example
	-------
	>>> from sluarch import spindles_rms
	>>> raw = mne.io.read_raw_edf('myfile.edf', preload=True)
	>>> spindles_rms(raw, sf=raw.info['sfreq'])

	Parameters
	----------
	raw :
		An MNE Raw instance.  The first EEG channel is used.
	sf :
		Sampling frequency in Hz.
	n_sd :
		Number of standard deviations above the mean RMS used as the
		detection threshold (default ``5.0``).
	freq_sigma :
		Sigma band ``(low, high)`` in Hz.  Default ``(10, 15)``Hz.
	win_sec :
		RMS window length in seconds (default ``0.3``s).
	step_sec :
		Step between successive windows in seconds (default ``0.1``s).
	min_dur :
		Minimum spindle duration in seconds (default ``0.5``s).
	max_dur :
		Maximum spindle duration in seconds (default ``3.0``s).
	resample_to :
		Target sampling rate in Hz applied when ``sf`` exceeds this value
		(default ``512``Hz).

	Returns
	-------
	int
		Number of detected sleep spindles.
	"""
	_validate_raw(raw,sf)
	if not (0<freq_sigma[0]<freq_sigma[1]):
		raise ValueError('freq_sigma must satisfy 0 < low < high.')
	if n_sd<=0:
		raise ValueError(f"'n_sd' must be positive, got {n_sd}.")
	raw,sf=_maybe_resample(raw,sf,resample_to)
	freq_broad=(1.0,30.0)
	data=raw.get_data()[0]*1_000_000
	data=filter_data(data,sf,freq_broad[0],freq_broad[1],method='fir',verbose=False)
	nyquist=sf/2.0
	sos=signal.iirfilter(20,[freq_sigma[0]/nyquist,freq_sigma[1]/nyquist],btype='bandpass',output='sos',)
	data_sigma=signal.sosfiltfilt(sos,data)
	win_samp=max(1,int(round(win_sec*sf)))
	step_samp=max(1,int(round(step_sec*sf)))
	n=len(data_sigma)
	if win_samp>n:
		raise ValueError(f'win_sec ({win_sec} s) is longer than the recording ({n/sf:.2f} s).')
	views=_sliding_window(data_sigma,win_samp,step_samp)
	rms_env=np.sqrt(np.mean(views**2,axis=1))
	threshold=rms_env.mean()+n_sd*rms_env.std(ddof=1)
	n_wins=rms_env.shape[0]
	tt=np.clip((np.arange(n_wins)*step_samp+win_samp/2.0)/sf,0,(n-1)/sf)
	rms_interp=interp1d(tt,rms_env,kind='linear',bounds_error=False,fill_value=(rms_env[0],rms_env[-1]),assume_sorted=True,)(np.arange(n)/sf)
	mask=rms_interp>=threshold
	count=_label_events(mask,sf,min_dur=min_dur,max_dur=max_dur)
	text='spindle' if count==1 else 'spindles'
	logger.info('Moving RMS (threshold=%.4f µV, n_sd=%.1f): %d %s detected.',threshold,n_sd,count,text)
	print(f'Using moving RMS (mean + {n_sd}×SD): {count} {text}')
	return count


def spindles_cov(raw: mne.io.BaseRaw,sf: float,n_sd: float = 3.0,freq_sigma: tuple[
	float,float] = (10.0,15.0),win_sec: float = 0.3,step_sec: float = 0.1,min_dur: float = 0.5,max_dur: float = 3.0,resample_to: int = 512,) -> int:
	"""Detect sleep spindles using a moving covariance envelope.

	Computes the windowed covariance between the broadband-filtered signal and
	the sigma-filtered signal::

	    Cov(broad, sigma) = mean( (broad − mean_broad) × (sigma − mean_sigma) )

	High positive covariance indicates sigma-band oscillations coherently
	riding the broadband signal — the hallmark signature of a sleep spindle.
	The detection threshold is set adaptively as::

	    threshold = mean(Cov) + n_sd × std(Cov)

	Because covariance distributions are less heavy-tailed than power
	distributions, a lower ``n_sd`` (default ``3.0``) is appropriate compared
	with the RMS-based detector.

	Notes
	-----
	If you use this toolbox, please cite as follows:

	* Guillot S.J., (2024). SlµArch. GitHub, Zenodo.
	  https://doi.org/10.5281/zenodo.11204061

	Example
	-------
	>>> from sluarch import spindles_cov
	>>> raw = mne.io.read_raw_edf('myfile.edf', preload=True)
	>>> spindles_cov(raw, sf=raw.info['sfreq'])

	Parameters
	----------
	raw :
		An MNE Raw instance.  The first EEG channel is used.
	sf :
		Sampling frequency in Hz.
	n_sd :
		Number of standard deviations above the mean covariance used as the
		detection threshold (default ``3.0``).
	freq_sigma :
		Sigma band ``(low, high)`` in Hz.  Default ``(10, 15)``Hz.
	win_sec :
		Covariance window length in seconds (default ``0.3``s).
	step_sec :
		Step between successive windows in seconds (default ``0.1``s).
	min_dur :
		Minimum spindle duration in seconds (default ``0.5``s).
	max_dur :
		Maximum spindle duration in seconds (default ``3.0``s).
	resample_to :
		Target sampling rate in Hz applied when ``sf`` exceeds this value
		(default ``512``Hz).

	Returns
	-------
	int
		Number of detected sleep spindles.
	"""
	_validate_raw(raw,sf)
	if not (0<freq_sigma[0]<freq_sigma[1]):
		raise ValueError('freq_sigma must satisfy 0 < low < high.')
	if n_sd<=0:
		raise ValueError(f"'n_sd' must be positive, got {n_sd}.")
	raw,sf=_maybe_resample(raw,sf,resample_to)
	freq_broad=(1.0,30.0)
	data=raw.get_data()[0]*1_000_000
	data_broad=filter_data(data,sf,freq_broad[0],freq_broad[1],method='fir',verbose=False)
	nyquist=sf/2.0
	sos=signal.iirfilter(20,[freq_sigma[0]/nyquist,freq_sigma[1]/nyquist],btype='bandpass',output='sos',)
	data_sigma=signal.sosfiltfilt(sos,data_broad)
	win_samp=max(1,int(round(win_sec*sf)))
	step_samp=max(1,int(round(step_sec*sf)))
	n=len(data_broad)
	if win_samp>n:
		raise ValueError(f'win_sec ({win_sec} s) is longer than the recording ({n/sf:.2f} s).')
	views_broad=_sliding_window(data_broad,win_samp,step_samp)
	views_sigma=_sliding_window(data_sigma,win_samp,step_samp)
	broad_demean=views_broad-views_broad.mean(axis=1,keepdims=True)
	sigma_demean=views_sigma-views_sigma.mean(axis=1,keepdims=True)
	cov_env=np.mean(broad_demean*sigma_demean,axis=1)
	threshold=cov_env.mean()+n_sd*cov_env.std(ddof=1)
	n_wins=cov_env.shape[0]
	tt=np.clip((np.arange(n_wins)*step_samp+win_samp/2.0)/sf,0,(n-1)/sf)
	cov_interp=interp1d(tt,cov_env,kind='linear',bounds_error=False,fill_value=(cov_env[0],cov_env[-1]),assume_sorted=True,)(np.arange(n)/sf)
	mask=cov_interp>=threshold
	count=_label_events(mask,sf,min_dur=min_dur,max_dur=max_dur)
	text='spindle' if count==1 else 'spindles'
	logger.info('Moving covariance (threshold=%.6f µV², n_sd=%.1f): %d %s detected.',threshold,n_sd,count,text)
	print(f'Using moving covariance (mean + {n_sd}×SD): {count} {text}')
	return count


# ---------------------------------------------------------------------------
# SLOW OSCILLATIONS
# ---------------------------------------------------------------------------


def slowosc(raw: mne.io.BaseRaw,sf: float,freq_so: tuple[float,float] = (0.5,2.0),freq_broad: tuple[float,float] = (0.1,100.0),resample_to: int = 512,) -> tuple[float,float,float]:
	"""Detect slow oscillations via Welch PSD area-under-curve ratio.

	The slow oscillation score is the ratio of SO-band AUC to broadband AUC
	computed from Welch's power spectral density estimate.

	Notes
	----------
	If you use this toolbox, please cite as follows:

	* Guillot S.J., (2024). SlµArch. GitHub, Zenodo.
	  https://doi.org/10.5281/zenodo.11204061

	Example
	-------
	>>> from sluarch import slowosc
	>>> raw = mne.io.read_raw_edf('myfile.edf', preload=True)
	>>> auc_sig, auc_psd, sos = slowosc(raw, sf=raw.info['sfreq'])

	Parameters
	----------
	raw :
		An MNE Raw instance.  The first EEG channel is used.
	sf :
		Sampling frequency in Hz.
	freq_so :
		Slow oscillation band ``(low, high)`` in Hz.
		Default ``(0.5, 2.0)``Hz.
	freq_broad :
		Broadband reference range ``(low, high)`` in Hz.
		Default ``(0.1, 100.0)``Hz.
	resample_to :
		Target sampling rate in Hz applied when ``sf`` exceeds this value
		(default ``512``Hz).

	Returns
	-------
	auc_sig : float
		Area-under-curve of the SO band PSD.
	auc_psd : float
		Area-under-curve of the broadband PSD.
	sos : float
		Slow oscillation score: ``auc_sig / auc_psd``.
	"""
	_validate_raw(raw,sf)
	if not (0<freq_so[0]<freq_so[1]):
		raise ValueError('freq_so must satisfy 0 < low < high.')
	if not (0<freq_broad[0]<freq_broad[1]):
		raise ValueError('freq_broad must satisfy 0 < low < high.')
	raw,sf=_maybe_resample(raw,sf,resample_to)
	filt_high=min(freq_broad[1],sf/2.0-1.0)
	data=raw.get_data()[0]*1_000_000
	data_filt=filter_data(data,sf,freq_broad[0],filt_high,method='fir',verbose=False)
	win=int(4*sf)
	freq,psd=signal.welch(data_filt,sf,nperseg=win)
	freq_res=freq[1]-freq[0]
	idx_so=np.logical_and(freq>=freq_so[0],freq<=freq_so[1])
	idx_bd=np.logical_and(freq>=freq_broad[0],freq<=freq_broad[1])
	auc_sig=float(simpson(psd[idx_so],dx=freq_res))
	auc_psd=float(simpson(psd[idx_bd],dx=freq_res))
	sos=auc_sig/auc_psd if auc_psd>0 else 0.0
	logger.info('Slow oscillation score: %.6f  (auc_sig=%.6f, auc_psd=%.6f)',sos,auc_sig,auc_psd)
	return auc_sig,auc_psd,sos


# ---------------------------------------------------------------------------
# K-COMPLEX
# ---------------------------------------------------------------------------


def kcomp(raw: mne.io.BaseRaw,sf: float,Kplot: bool = False,resample_to: int = 512,) -> int:
	"""Detect K-complexes.

	Spectrogram power in the 0.3–1 Hz band is extracted and peaks exceeding
	the mean level are labelled as K-complex events.  When ``Kplot=True``
	each detected event is visualised using MNE Epochs.

	Notes
	----------
	If you use this toolbox, please cite as follows:

	* Guillot S.J., (2024). SlµArch. GitHub, Zenodo.
	  https://doi.org/10.5281/zenodo.11204061

	Example
	-------
	>>> from sluarch import kcomp
	>>> raw = mne.io.read_raw_edf('myfile.edf', preload=True)
	>>> n_kc = kcomp(raw, sf=raw.info['sfreq'])

	Parameters
	----------
	raw :
		An MNE Raw instance.  The first EEG channel is used.
	sf :
		Sampling frequency in Hz.
	Kplot :
		If ``True``, plot all detected K-complex epochs with MNE.
		Default ``False``.
	resample_to :
		Target sampling rate in Hz applied when ``sf`` exceeds this value
		(default ``512``Hz).

	Returns
	-------
	int
		Number of detected K-complexes.
	"""
	_validate_raw(raw,sf)
	raw_proc,sf_proc=_maybe_resample(raw,sf,resample_to)
	freq_broad=(1.0,45.0)
	data=raw_proc.get_data()[0]*1_000_000
	data_filt=filter_data(data,sf_proc,freq_broad[0],freq_broad[1],method='fir',verbose=False)
	f,_t,Sxx=signal.spectrogram(data_filt,sf_proc,mode='psd',scaling='density')
	freq_slice=(f>=0.3)&(f<=1.0)
	Sxx_K=Sxx[freq_slice,:].mean(axis=0)
	K_threshold=Sxx_K.mean()
	K_spikes=Sxx_K>=K_threshold
	labl,nbK=ndimage.label(K_spikes)
	if Kplot:
		start_list=[]
		for label in range(1,nbK+1):
			indices=np.where(labl==label)[0]
			start_list.append(int(indices[0]))
		K_events=np.zeros((len(start_list),3),dtype=int)
		K_events[:,0]=start_list
		K_events[:,2]=1
		K_epochs=mne.Epochs(raw_proc,K_events,event_id={'K_complex':1},baseline=None,tmin=0,tmax=1,preload=True,verbose=False,)
		K_epochs.plot()
	logger.info('K-complexes detected: %d',nbK)
	return nbK
