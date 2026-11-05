# Copyright 2026 Simon J. Guillot
# SPDX-License-Identifier: Apache-2.0

from itertools import pairwise
from typing import Literal

import mne
import numpy as np
from scipy import ndimage, signal
from scipy.integrate import simpson

__all__ = ['detect_kcomp', 'detect_slowosc', 'detect_spindles']

SpindleMethod = Literal['auto', 'fixed']


def _validate_raw(raw: mne.io.BaseRaw, sf: float) -> None:
	"""Validate an MNE recording and its supplied sampling frequency."""
	if not isinstance(raw, mne.io.BaseRaw):
		raise TypeError(f'raw must be an MNE BaseRaw instance, got {type(raw).__name__!r}.')
	if not isinstance(sf, (int, float)) or not np.isfinite(sf) or sf <= 0:
		raise ValueError(f'sf must be a positive finite sampling frequency, got {sf}.')
	actual_sf = float(raw.info['sfreq'])
	if not np.isclose(sf, actual_sf):
		raise ValueError(f'sf ({sf}) does not match raw.info["sfreq"] ({actual_sf}).')
	if not raw.ch_names:
		raise ValueError('raw must contain at least one channel.')


def _validate_band(band: tuple[float, float], sf: float, name: str) -> None:
	"""Validate a frequency band against the Nyquist frequency."""
	if len(band) != 2 or not all(np.isfinite(value) for value in band):
		raise ValueError(f'{name} must contain two finite frequencies.')
	low, high = band
	if not 0 < low < high < sf / 2:
		raise ValueError(f'{name} must satisfy 0 < low < high < Nyquist ({sf / 2:g} Hz).')


def _validate_range(bounds: tuple[float, float], name: str) -> None:
	"""Validate a positive inclusive range."""
	if len(bounds) != 2 or not all(np.isfinite(value) for value in bounds):
		raise ValueError(f'{name} must contain two finite values.')
	if not 0 < bounds[0] <= bounds[1]:
		raise ValueError(f'{name} must satisfy 0 < minimum <= maximum.')


def _prepare_signal(
	raw: mne.io.BaseRaw, sf: float | None, pick: int | str, resample_to: int | None
) -> tuple[np.ndarray, float, float, int]:
	"""Return one selected channel in microvolts, optionally resampled."""
	if sf is None:
		sf = float(raw.info['sfreq'])
	_validate_raw(raw, sf)
	if resample_to is not None and (
		not isinstance(resample_to, (int, float)) or not np.isfinite(resample_to) or resample_to <= 0
	):
		raise ValueError(f'resample_to must be positive or None, got {resample_to}.')
	original_sf = float(sf)
	original_n = raw.n_times
	work = raw
	if resample_to is not None and sf > resample_to:
		work = raw.copy().resample(resample_to, verbose=False)
		sf = float(work.info['sfreq'])
	try:
		data = work.get_data(picks=[pick]).ravel() * 1_000_000
	except (IndexError, ValueError) as error:
		raise ValueError(f'could not select EEG channel {pick!r}.') from error
	if data.size == 0:
		raise ValueError('raw contains no samples.')
	return data, float(sf), original_sf, original_n


def _resample_mask(mask: np.ndarray | None, original_n: int, working_n: int, name: str) -> np.ndarray | None:
	"""Nearest-neighbour resample a sample-level boolean mask."""
	if mask is None:
		return None
	mask = np.asarray(mask, dtype=bool).ravel()
	if mask.size == working_n:
		return mask
	if mask.size != original_n:
		raise ValueError(f'{name} must contain {original_n} or {working_n} samples, got {mask.size}.')
	indices = np.minimum((np.arange(working_n) * original_n / working_n).astype(np.int64), original_n - 1)
	return mask[indices]


def _valid_mask(
	nrem_mask: np.ndarray | None, artifact_mask: np.ndarray | None, original_n: int, working_n: int
) -> tuple[np.ndarray, np.ndarray | None]:
	"""Build a valid detection mask and return the resampled NREM mask."""
	nrem = _resample_mask(nrem_mask, original_n, working_n, 'nrem_mask')
	artifact = _resample_mask(artifact_mask, original_n, working_n, 'artifact_mask')
	valid = np.ones(working_n, dtype=bool)
	if nrem is not None:
		valid &= nrem
	if artifact is not None:
		valid &= ~artifact
	return valid, nrem


def _event_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Return start-inclusive and stop-exclusive indices of true runs."""
	mask = np.asarray(mask, dtype=bool).ravel()
	edges = np.diff(np.pad(mask.view(np.int8), (1, 1)))
	return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _adaptive_spindle_events(
	data: np.ndarray,
	sf: float,
	freq_sigma: tuple[float, float],
	valid: np.ndarray,
	baseline: np.ndarray,
	rms_win_sec: float,
	lower_multiplier: float,
	upper_multiplier: float,
	min_dur: float,
	max_dur: float,
	min_isi: float,
) -> tuple[list[dict], dict]:
	"""Run the dual-threshold cubed-RMS detector at the working rate."""
	sos = signal.butter(4, freq_sigma, btype='bandpass', fs=sf, output='sos')
	sigma = signal.sosfiltfilt(sos, data)
	win = max(1, round(rms_win_sec * sf))
	rms = np.sqrt(ndimage.uniform_filter1d(np.square(sigma), size=win, mode='nearest'))
	transformed = np.power(rms, 3)
	if not baseline.any():
		raise ValueError('no valid samples are available for threshold estimation.')
	baseline_mean = float(np.mean(transformed[baseline]))
	lower = lower_multiplier * baseline_mean
	upper = upper_multiplier * baseline_mean
	starts, stops = _event_runs((transformed > lower) & valid)
	candidates = [
		[int(start), int(stop)]
		for start, stop in zip(starts, stops, strict=True)
		if np.max(transformed[start:stop], initial=0) > upper
	]
	merged: list[list[int]] = []
	max_gap = round(min_isi * sf)
	for start, stop in candidates:
		if merged and start - merged[-1][1] <= max_gap and np.all(valid[merged[-1][1] : start]):
			merged[-1][1] = stop
		else:
			merged.append([start, stop])
	events = []
	for start, stop in merged:
		duration = (stop - start) / sf
		if not min_dur <= duration <= max_dur:
			continue
		segment = sigma[start:stop]
		frequencies, power = signal.periodogram(segment, fs=sf, detrend='constant')
		in_band = (frequencies >= freq_sigma[0]) & (frequencies <= freq_sigma[1])
		dominant = float('nan')
		if np.any(in_band) and np.any(power[in_band] > 0):
			dominant = float(frequencies[in_band][np.argmax(power[in_band])])
		peak = start + int(np.argmax(np.abs(segment)))
		events.append(
			{
				'_start_working': start,
				'_stop_working': stop,
				'_peak_working': peak,
				'duration': float(duration),
				'rms_peak_uv': float(np.max(rms[start:stop])),
				'sigma_peak_uv': float(np.max(np.abs(segment))),
				'dominant_frequency_hz': dominant,
			}
		)
	return events, {
		'baseline_mean_cubed_rms': baseline_mean,
		'lower': lower,
		'upper': upper,
		'lower_multiplier': lower_multiplier,
		'upper_multiplier': upper_multiplier,
	}


def _map_spindle_events(
	events: list[dict], working_sf: float, original_sf: float, original_n: int, spindle_type: str | None
) -> list[dict]:
	"""Map private working indices to public original-recording indices."""
	mapped = []
	for source in events:
		start = source.pop('_start_working')
		stop = source.pop('_stop_working')
		peak = source.pop('_peak_working')
		mapped.append(
			{
				'start': round(start * original_sf / working_sf),
				'end': min(original_n, round(stop * original_sf / working_sf)),
				'peak': min(original_n - 1, round(peak * original_sf / working_sf)),
				'start_sec': float(start / working_sf),
				'end_sec': float(stop / working_sf),
				'peak_sec': float(peak / working_sf),
				**source,
				'spindle_type': spindle_type,
			}
		)
	return mapped


def _spindles_adaptenv(
	raw: mne.io.BaseRaw,
	sf: float | None = None,
	pick: int | str = 0,
	freq_sigma: tuple[float, float] | None = None,
	slow_band: tuple[float, float] = (10.0, 12.0),
	fast_band: tuple[float, float] = (12.5, 16.0),
	rms_win_sec: float = 0.75,
	lower_multiplier: float = 1.2,
	upper_multiplier: float = 3.5,
	min_dur: float = 0.5,
	max_dur: float = 10.0,
	min_isi: float = 0.1,
	resample_to: int | None = 512,
	nrem_mask: np.ndarray | None = None,
	artifact_mask: np.ndarray | None = None,
	threshold_from_nrem: bool = False,
	classification_split_hz: float | None = None,
) -> dict:
	"""Detect slow and fast rodent spindles with an adaptive RMS envelope.

	The selected EEG channel is sigma filtered, converted to a smoothed RMS
	envelope, and cubed. A lower threshold defines event boundaries and an
	upper threshold confirms candidates, applied separately to the
	10–12Hz slow and 12.5–16Hz fast bands. Pass ``freq_sigma`` to run one
	known band instead.
	"""
	_validate_range((min_dur, max_dur), 'duration')
	if rms_win_sec <= 0 or min_isi < 0:
		raise ValueError('rms_win_sec must be positive and min_isi must be non-negative.')
	if not 0 < lower_multiplier < upper_multiplier:
		raise ValueError('threshold multipliers must satisfy 0 < lower_multiplier < upper_multiplier.')
	data, work_sf, original_sf, original_n = _prepare_signal(raw, sf, pick, resample_to)
	valid, nrem = _valid_mask(nrem_mask, artifact_mask, original_n, data.size)
	baseline = valid.copy()
	if threshold_from_nrem:
		if nrem is None:
			raise ValueError('threshold_from_nrem=True requires nrem_mask.')
		baseline &= nrem
	if freq_sigma is None:
		_validate_band(slow_band, work_sf, 'slow_band')
		_validate_band(fast_band, work_sf, 'fast_band')
		if slow_band[1] > fast_band[0]:
			raise ValueError('slow_band and fast_band must not overlap.')
		if classification_split_hz is not None:
			raise ValueError('classification_split_hz is only used with an explicit freq_sigma band.')
		all_events = []
		thresholds = {}
		for spindle_type, band in (('slow', slow_band), ('fast', fast_band)):
			events, band_thresholds = _adaptive_spindle_events(
				data,
				work_sf,
				band,
				valid,
				baseline,
				rms_win_sec,
				lower_multiplier,
				upper_multiplier,
				min_dur,
				max_dur,
				min_isi,
			)
			all_events.extend(_map_spindle_events(events, work_sf, original_sf, original_n, spindle_type))
			thresholds[spindle_type] = band_thresholds
		mapped = sorted(all_events, key=lambda event: event['start'])
		counts = {
			'slow': sum(event['spindle_type'] == 'slow' for event in mapped),
			'fast': sum(event['spindle_type'] == 'fast' for event in mapped),
			'unclassified': 0,
		}
		method = 'adaptive_envelope_separate_bands'
	else:
		_validate_band(freq_sigma, work_sf, 'freq_sigma')
		if classification_split_hz is not None and not (freq_sigma[0] < classification_split_hz < freq_sigma[1]):
			raise ValueError('classification_split_hz must lie strictly inside freq_sigma.')
		events, thresholds = _adaptive_spindle_events(
			data,
			work_sf,
			freq_sigma,
			valid,
			baseline,
			rms_win_sec,
			lower_multiplier,
			upper_multiplier,
			min_dur,
			max_dur,
			min_isi,
		)
		mapped = _map_spindle_events(events, work_sf, original_sf, original_n, None)
		counts = {'slow': 0, 'fast': 0, 'unclassified': 0}
		for event in mapped:
			frequency = event['dominant_frequency_hz']
			if classification_split_hz is None or not np.isfinite(frequency):
				counts['unclassified'] += 1
			elif frequency < classification_split_hz:
				event['spindle_type'] = 'slow'
				counts['slow'] += 1
			else:
				event['spindle_type'] = 'fast'
				counts['fast'] += 1
		method = 'adaptive_envelope_single_band'
	return {
		'count': len(mapped),
		'counts': {'total': len(mapped), **counts},
		'events': mapped,
		'thresholds': thresholds,
		'parameters': {
			'method': method,
			'sf_original': original_sf,
			'sf_working': work_sf,
			'pick': pick,
			'freq_sigma': freq_sigma,
			'slow_band': slow_band,
			'fast_band': fast_band,
			'rms_win_sec': rms_win_sec,
			'min_dur': min_dur,
			'max_dur': max_dur,
			'min_isi': min_isi,
			'threshold_from_nrem': threshold_from_nrem,
			'classification_split_hz': classification_split_hz,
		},
	}


def _individual_spindle_window(frequencies: np.ndarray, power: np.ndarray, search_band: tuple[float, float]) -> dict:
	"""Estimate an individual contiguous half-power window around a PSD peak."""
	in_search = (frequencies >= search_band[0]) & (frequencies <= search_band[1])
	if np.count_nonzero(in_search) < 3:
		raise ValueError(f'insufficient frequency resolution inside search band {search_band}.')
	freq = frequencies[in_search]
	curve = ndimage.gaussian_filter1d(power[in_search], sigma=1.0, mode='nearest')
	peak_index = int(np.argmax(curve))
	peak_frequency = float(freq[peak_index])
	baseline = float(np.median(curve))
	peak_power = float(curve[peak_index])
	half_height = baseline + 0.5 * max(0.0, peak_power - baseline)
	left = peak_index
	right = peak_index
	while left > 0 and curve[left - 1] >= half_height:
		left -= 1
	while right < curve.size - 1 and curve[right + 1] >= half_height:
		right += 1
	resolution = float(np.median(np.diff(freq)))
	half_width = max(0.5, (freq[right] - freq[left]) / 2 + resolution)
	window = (
		float(max(search_band[0], peak_frequency - half_width)),
		float(min(search_band[1], peak_frequency + half_width)),
	)
	prominence_ratio = (peak_power - baseline) / max(baseline, np.finfo(float).tiny)
	return {
		'peak_hz': peak_frequency,
		'window_hz': window,
		'peak_power': peak_power,
		'baseline_power': baseline,
		'prominence_ratio': float(prominence_ratio),
	}


def _spindles_multimethod(
	raw: mne.io.BaseRaw,
	sf: float | None = None,
	pick: int | str = 0,
	slow_search: tuple[float, float] = (10.0, 12.0),
	fast_search: tuple[float, float] = (12.5, 16.0),
	min_peak_prominence: float = 0.75,
	resample_to: int | None = 512,
	nrem_mask: np.ndarray | None = None,
	artifact_mask: np.ndarray | None = None,
	**adaptive_options,
) -> dict:
	"""Automatically detect and classify slow and fast rodent spindles.

	Individual slow and fast peak windows are estimated from the recording's
	NREM power spectrum. If both peaks are resolved, each dynamic band is
	detected separately. Otherwise, the function automatically falls back to
	one broad adaptive-envelope pass and classifies events by their dominant
	frequency.
	"""
	data, work_sf, original_sf, original_n = _prepare_signal(raw, sf, pick, resample_to)
	_validate_band(slow_search, work_sf, 'slow_search')
	_validate_band(fast_search, work_sf, 'fast_search')
	if slow_search[1] > fast_search[0]:
		raise ValueError('slow_search and fast_search must not overlap.')
	if min_peak_prominence < 0 or not np.isfinite(min_peak_prominence):
		raise ValueError('min_peak_prominence must be non-negative and finite.')
	reserved = {
		'artifact_mask',
		'classification_split_hz',
		'fast_band',
		'freq_sigma',
		'nrem_mask',
		'pick',
		'raw',
		'resample_to',
		'sf',
		'slow_band',
		'threshold_from_nrem',
	}
	if conflicts := reserved.intersection(adaptive_options):
		raise ValueError(f'adaptive_options cannot override: {", ".join(sorted(conflicts))}.')
	valid, nrem = _valid_mask(nrem_mask, artifact_mask, original_n, data.size)
	psd_mask = valid if nrem is None else valid & nrem
	if np.count_nonzero(psd_mask) < round(2 * work_sf):
		raise ValueError('at least two seconds of valid EEG are required to estimate spindle bands.')
	psd_data = signal.detrend(data[psd_mask], type='constant')
	nperseg = min(round(5 * work_sf), psd_data.size)
	frequencies, power = signal.welch(psd_data, work_sf, nperseg=nperseg)
	slow_spectrum = _individual_spindle_window(frequencies, power, slow_search)
	fast_spectrum = _individual_spindle_window(frequencies, power, fast_search)
	peaks_resolved = (
		slow_spectrum['prominence_ratio'] >= min_peak_prominence
		and fast_spectrum['prominence_ratio'] >= min_peak_prominence
		and fast_spectrum['peak_hz'] - slow_spectrum['peak_hz'] >= 0.75
	)
	common = {
		'raw': raw,
		'sf': float(raw.info['sfreq']) if sf is None else sf,
		'pick': pick,
		'resample_to': resample_to,
		'nrem_mask': nrem_mask,
		'artifact_mask': artifact_mask,
		'threshold_from_nrem': nrem_mask is not None,
		**adaptive_options,
	}
	if peaks_resolved:
		slow_result = _spindles_adaptenv(freq_sigma=slow_spectrum['window_hz'], **common)
		fast_result = _spindles_adaptenv(freq_sigma=fast_spectrum['window_hz'], **common)
		for event in slow_result['events']:
			event['spindle_type'] = 'slow'
		for event in fast_result['events']:
			event['spindle_type'] = 'fast'
		events = sorted([*slow_result['events'], *fast_result['events']], key=lambda event: event['start'])
		method = 'separate_dynamic_bands'
		thresholds = {'slow': slow_result['thresholds'], 'fast': fast_result['thresholds']}
	else:
		broad_band = (slow_search[0], fast_search[1])
		result = _spindles_adaptenv(freq_sigma=broad_band, **common)
		events = result['events']
		for event in events:
			frequency = event['dominant_frequency_hz']
			if slow_search[0] <= frequency <= slow_search[1]:
				event['spindle_type'] = 'slow'
			elif fast_search[0] <= frequency <= fast_search[1]:
				event['spindle_type'] = 'fast'
		method = 'broad_band_dominant_frequency'
		thresholds = {'broad': result['thresholds']}
	counts = {
		'slow': sum(event['spindle_type'] == 'slow' for event in events),
		'fast': sum(event['spindle_type'] == 'fast' for event in events),
		'unclassified': sum(event['spindle_type'] is None for event in events),
	}
	return {
		'count': len(events),
		'counts': {'total': len(events), **counts},
		'events': events,
		'method_selected': method,
		'spectral_peaks': {'slow': slow_spectrum, 'fast': fast_spectrum},
		'thresholds': thresholds,
		'parameters': {
			'sf_original': original_sf,
			'sf_working': work_sf,
			'pick': pick,
			'slow_search': slow_search,
			'fast_search': fast_search,
			'min_peak_prominence': min_peak_prominence,
		},
	}


def detect_slowosc(
	raw: mne.io.BaseRaw,
	sf: float | None = None,
	pick: int | str = 0,
	filter_band: tuple[float, float] = (0.1, 4.0),
	negative_halfwave: tuple[float, float] = (0.3, 1.0),
	negative_threshold_uv: float = -80.0,
	peak_to_peak_threshold_uv: float = 140.0,
	freq_so: tuple[float, float] = (0.5, 2.0),
	freq_broad: tuple[float, float] = (0.1, 100.0),
	resample_to: int | None = 256,
	nrem_mask: np.ndarray | None = None,
	artifact_mask: np.ndarray | None = None,
) -> dict:
	"""Detect slow oscillations and return waveform and spectral measures.

	Candidates follow the established Massimini morphology: a
	positive-to-negative zero crossing, a 0.3–1.0s negative half-wave,
	a trough of at most -80µV, and at least 140µV peak-to-peak amplitude.
	Thresholds are configurable for species, montage, and reference polarity.
	"""
	_validate_range(negative_halfwave, 'negative_halfwave')
	if not np.isfinite(negative_threshold_uv) or negative_threshold_uv >= 0:
		raise ValueError('negative_threshold_uv must be a finite negative value.')
	if not np.isfinite(peak_to_peak_threshold_uv) or peak_to_peak_threshold_uv <= 0:
		raise ValueError('peak_to_peak_threshold_uv must be positive and finite.')
	data, work_sf, original_sf, original_n = _prepare_signal(raw, sf, pick, resample_to)
	_validate_band(filter_band, work_sf, 'filter_band')
	_validate_band(freq_so, work_sf, 'freq_so')
	if not 0 < freq_broad[0] < freq_broad[1] or freq_broad[0] >= work_sf / 2:
		raise ValueError('freq_broad must satisfy 0 < low < high and begin below Nyquist.')
	valid, _ = _valid_mask(nrem_mask, artifact_mask, original_n, data.size)
	sos_filter = signal.butter(4, filter_band, btype='bandpass', fs=work_sf, output='sos')
	filtered = signal.sosfiltfilt(sos_filter, data)
	positive_to_negative = np.flatnonzero((filtered[:-1] >= 0) & (filtered[1:] < 0)) + 1
	negative_to_positive = np.flatnonzero((filtered[:-1] < 0) & (filtered[1:] >= 0)) + 1
	events = []
	for start_value, end_value in pairwise(positive_to_negative):
		start = int(start_value)
		end = int(end_value)
		middle_candidates = negative_to_positive[(negative_to_positive > start) & (negative_to_positive < end)]
		if middle_candidates.size == 0 or not np.all(valid[start:end]):
			continue
		middle = int(middle_candidates[0])
		negative_duration = (middle - start) / work_sf
		if not negative_halfwave[0] <= negative_duration <= negative_halfwave[1]:
			continue
		trough = start + int(np.argmin(filtered[start:middle]))
		peak = middle + int(np.argmax(filtered[middle:end]))
		trough_uv = float(filtered[trough])
		peak_uv = float(filtered[peak])
		peak_to_peak_uv = peak_uv - trough_uv
		if trough_uv > negative_threshold_uv or peak_to_peak_uv < peak_to_peak_threshold_uv:
			continue
		events.append(
			{
				'start': round(start * original_sf / work_sf),
				'end': min(original_n, round(end * original_sf / work_sf)),
				'trough': min(original_n - 1, round(trough * original_sf / work_sf)),
				'peak': min(original_n - 1, round(peak * original_sf / work_sf)),
				'start_sec': float(start / work_sf),
				'end_sec': float(end / work_sf),
				'trough_sec': float(trough / work_sf),
				'peak_sec': float(peak / work_sf),
				'duration': float((end - start) / work_sf),
				'negative_halfwave_duration': float(negative_duration),
				'trough_uv': trough_uv,
				'peak_uv': peak_uv,
				'peak_to_peak_uv': peak_to_peak_uv,
			}
		)
	win = min(round(4 * work_sf), data.size)
	if win < 2:
		raise ValueError('raw is too short to estimate a power spectrum.')
	frequencies, psd = signal.welch(data, work_sf, nperseg=win)
	resolution = frequencies[1] - frequencies[0]
	broad_high = min(freq_broad[1], work_sf / 2)
	in_so = (frequencies >= freq_so[0]) & (frequencies <= freq_so[1])
	in_broad = (frequencies >= freq_broad[0]) & (frequencies <= broad_high)
	auc_signal = float(simpson(psd[in_so], dx=resolution))
	auc_total = float(simpson(psd[in_broad], dx=resolution))
	return {
		'count': len(events),
		'events': events,
		'spectral': {
			'auc_signal': auc_signal,
			'auc_total': auc_total,
			'score': auc_signal / auc_total if auc_total > 0 else 0.0,
		},
		'parameters': {
			'sf_original': original_sf,
			'sf_working': work_sf,
			'pick': pick,
			'filter_band': filter_band,
			'negative_halfwave': negative_halfwave,
			'negative_threshold_uv': negative_threshold_uv,
			'peak_to_peak_threshold_uv': peak_to_peak_threshold_uv,
		},
	}


def detect_kcomp(
	raw: mne.io.BaseRaw,
	sf: float | None = None,
	pick: int | str = 0,
	filter_band: tuple[float, float] = (0.3, 4.0),
	duration: tuple[float, float] = (0.48, 1.5),
	peak_to_peak_threshold_uv: float = 50.0,
	resample_to: int | None = 256,
	nrem_mask: np.ndarray | None = None,
	artifact_mask: np.ndarray | None = None,
	Kplot: bool = False,
) -> dict:
	"""Detect biphasic K-complex candidates and return their morphology.

	Candidates contain a negative component followed by a positive rebound,
	last 0.48–1.5s, and span at least 50µV peak-to-peak by default.
	Use an N2 mask when possible: one EEG channel cannot distinguish an
	isolated K-complex from every morphologically similar slow wave.
	"""
	_validate_range(duration, 'duration')
	if not np.isfinite(peak_to_peak_threshold_uv) or peak_to_peak_threshold_uv <= 0:
		raise ValueError('peak_to_peak_threshold_uv must be positive and finite.')
	if not isinstance(Kplot, (bool, np.bool_)):
		raise TypeError('Kplot must be a boolean.')
	data, work_sf, original_sf, original_n = _prepare_signal(raw, sf, pick, resample_to)
	_validate_band(filter_band, work_sf, 'filter_band')
	valid, _ = _valid_mask(nrem_mask, artifact_mask, original_n, data.size)
	sos_filter = signal.butter(4, filter_band, btype='bandpass', fs=work_sf, output='sos')
	filtered = signal.sosfiltfilt(sos_filter, data)
	positive_to_negative = np.flatnonzero((filtered[:-1] >= 0) & (filtered[1:] < 0)) + 1
	negative_to_positive = np.flatnonzero((filtered[:-1] < 0) & (filtered[1:] >= 0)) + 1
	events = []
	for start_value, end_value in pairwise(positive_to_negative):
		start = int(start_value)
		end = int(end_value)
		event_duration = (end - start) / work_sf
		if not duration[0] <= event_duration <= duration[1] or not np.all(valid[start:end]):
			continue
		middle_candidates = negative_to_positive[(negative_to_positive > start) & (negative_to_positive < end)]
		if middle_candidates.size == 0:
			continue
		middle = int(middle_candidates[0])
		trough = start + int(np.argmin(filtered[start:middle]))
		peak = middle + int(np.argmax(filtered[middle:end]))
		trough_uv = float(filtered[trough])
		peak_uv = float(filtered[peak])
		peak_to_peak_uv = peak_uv - trough_uv
		if trough_uv >= 0 or peak_uv <= 0 or peak_to_peak_uv < peak_to_peak_threshold_uv:
			continue
		events.append(
			{
				'start': round(start * original_sf / work_sf),
				'end': min(original_n, round(end * original_sf / work_sf)),
				'trough': min(original_n - 1, round(trough * original_sf / work_sf)),
				'peak': min(original_n - 1, round(peak * original_sf / work_sf)),
				'start_sec': float(start / work_sf),
				'end_sec': float(end / work_sf),
				'trough_sec': float(trough / work_sf),
				'peak_sec': float(peak / work_sf),
				'duration': float(event_duration),
				'trough_uv': trough_uv,
				'peak_uv': peak_uv,
				'peak_to_peak_uv': peak_to_peak_uv,
			}
		)
	if Kplot and events:
		mne_events = np.zeros((len(events), 3), dtype=int)
		mne_events[:, 0] = [event['start'] for event in events]
		mne_events[:, 2] = 1
		epochs = mne.Epochs(
			raw, mne_events, event_id={'K_complex': 1}, baseline=None, tmin=-0.2, tmax=1.5, preload=True, verbose=False
		)
		epochs.plot()
	return {
		'count': len(events),
		'events': events,
		'parameters': {
			'sf_original': original_sf,
			'sf_working': work_sf,
			'pick': pick,
			'filter_band': filter_band,
			'duration': duration,
			'peak_to_peak_threshold_uv': peak_to_peak_threshold_uv,
		},
	}


def detect_spindles(raw: mne.io.BaseRaw, sf: float | None = None, *, method: SpindleMethod = 'auto', **kwargs) -> dict:
	"""Detect spindles using a selected SlµArch calculation method.

	Parameters
	----------
	method:
		``'auto'`` estimates recording-specific spindle bands and selects
		separate-band or broad-band detection. ``'fixed'`` uses the configured
		fixed slow and fast bands directly.
	**kwargs:
		Options forwarded to the selected detector.
	"""
	detectors = {'fixed': _spindles_adaptenv, 'auto': _spindles_multimethod}
	try:
		detector = detectors[method]
	except (KeyError, TypeError) as error:
		choices = ', '.join(repr(choice) for choice in detectors)
		raise ValueError(f'method must be one of {choices}; got {method!r}.') from error
	result = detector(raw, sf=sf, **kwargs)
	result['parameters']['method_requested'] = method
	return result
