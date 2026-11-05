"""Tests for the SlµArch public API."""

import mne
import numpy as np
import pytest

import sluarch


def _raw(data_uv: np.ndarray,sf: float = 100.0) -> mne.io.RawArray:
	info=mne.create_info(['EEG'],sf,'eeg')
	return mne.io.RawArray(np.asarray(data_uv)[np.newaxis]*1e-6,info,verbose=False)


@pytest.fixture(scope='module')
def mixed_spindle_raw() -> tuple[mne.io.RawArray,float]:
	"""Return deterministic rodent-like EEG with slow and fast spindles."""
	sf=100.0
	times=np.arange(round(60*sf))/sf
	rng=np.random.default_rng(4)
	data=rng.normal(0,0.3,times.size)
	for start in (5,25,45):
		burst=(times>=start)&(times<start+1)
		data[burst]+=30*np.sin(2*np.pi*10.5*times[burst])
	for start in (15,35,52):
		burst=(times>=start)&(times<start+1)
		data[burst]+=30*np.sin(2*np.pi*14*times[burst])
	return _raw(data,sf),sf


def test_public_api_is_consolidated() -> None:
	assert sluarch.__all__==['detect_kcomp','detect_slowosc','detect_spindles']
	assert all(callable(getattr(sluarch,name)) for name in sluarch.__all__)


def test_adaptive_envelope_detects_and_classifies_spindles(mixed_spindle_raw: tuple[mne.io.RawArray,float]) -> None:
	raw,sf=mixed_spindle_raw
	result=sluarch.detect_spindles(raw,sf,method='fixed')
	assert result['count']==6
	assert result['counts']['slow']==3
	assert result['counts']['fast']==3
	assert result['parameters']['slow_band']==(10,12)
	assert result['parameters']['fast_band']==(12.5,16)
	assert all(event['start']<event['peak']<event['end'] for event in result['events'])


def test_multimethod_uses_individual_dynamic_bands(mixed_spindle_raw: tuple[mne.io.RawArray,float]) -> None:
	raw,sf=mixed_spindle_raw
	result=sluarch.detect_spindles(raw,sf,method='auto')
	assert result['method_selected']=='separate_dynamic_bands'
	assert result['counts']['slow']==3
	assert result['counts']['fast']==3
	assert result['spectral_peaks']['slow']['peak_hz']==pytest.approx(10.5,abs=0.3)
	assert result['spectral_peaks']['fast']['peak_hz']==pytest.approx(14.0,abs=0.3)


def test_multimethod_automatically_falls_back_to_broad_band(mixed_spindle_raw: tuple[mne.io.RawArray,float]) -> None:
	raw,sf=mixed_spindle_raw
	result=sluarch.detect_spindles(raw,sf,method='auto',min_peak_prominence=1e9)
	assert result['method_selected']=='broad_band_dominant_frequency'
	assert result['count']==6
	assert result['counts']['slow']==3
	assert result['counts']['fast']==3


def test_artifact_mask_excludes_spindle(mixed_spindle_raw: tuple[mne.io.RawArray,float]) -> None:
	raw,sf=mixed_spindle_raw
	artifact=np.zeros(raw.n_times,dtype=bool)
	artifact[450:650]=True
	result=sluarch.detect_spindles(raw,sf,method='fixed',artifact_mask=artifact)
	assert result['count']==5


def test_slowosc_returns_physiological_cycles_and_spectral_score() -> None:
	sf=100.0
	times=np.arange(round(30*sf))/sf
	rng=np.random.default_rng(9)
	data=100*np.sin(2*np.pi*times)+rng.normal(0,1,times.size)
	result=sluarch.detect_slowosc(_raw(data,sf),sf)
	assert result['count']>=20
	assert result['spectral']['score']==pytest.approx(0.8,abs=0.02)
	assert all(event['trough_uv']<=-80 for event in result['events'])
	assert all(event['peak_to_peak_uv']>=140 for event in result['events'])
	assert all(0.3<=event['negative_halfwave_duration']<=1.0 for event in result['events'])


def test_kcomp_returns_three_biphasic_waveforms() -> None:
	sf=100.0
	times=np.arange(round(30*sf))/sf
	rng=np.random.default_rng(2)
	data=rng.normal(0,2,times.size)
	for centre in (5,15,24):
		data+=-120*np.exp(-0.5*((times-centre)/0.10)**2)
		data+=80*np.exp(-0.5*((times-centre-0.45)/0.18)**2)
	result=sluarch.detect_kcomp(_raw(data,sf),sf)
	assert result['count']==3
	assert [event['trough_sec'] for event in result['events']]==pytest.approx([5,15,24],abs=0.05)
	assert all(0.48<=event['duration']<=1.5 for event in result['events'])
	assert all(event['peak_to_peak_uv']>=50 for event in result['events'])


@pytest.mark.parametrize('detector',[sluarch.detect_spindles,sluarch.detect_kcomp])
def test_event_detectors_do_not_report_flat_signal(detector: object) -> None:
	sf=100.0
	result=detector(_raw(np.zeros(round(10*sf)),sf),sf)
	assert result['count']==0


@pytest.mark.parametrize('sf',[0,-1,np.nan])
def test_invalid_sampling_frequency_is_rejected(sf: float) -> None:
	raw=_raw(np.zeros(1_000))
	with pytest.raises(ValueError,match='positive finite'):
		sluarch.detect_spindles(raw,sf,method='fixed')


def test_sampling_frequency_must_match_raw() -> None:
	raw=_raw(np.zeros(1_000))
	with pytest.raises(ValueError,match='does not match'):
		sluarch.detect_slowosc(raw,101)


def test_frequency_bands_must_be_below_nyquist() -> None:
	raw=_raw(np.zeros(1_000))
	with pytest.raises(ValueError,match='Nyquist'):
		sluarch.detect_spindles(raw,100,method='fixed',freq_sigma=(10,50))


def test_unified_detector_method_dispatch(mixed_spindle_raw: tuple[mne.io.RawArray,float]) -> None:
	raw,sf=mixed_spindle_raw
	adaptive=sluarch.detect_spindles(raw,sf,method='fixed')
	multimethod=sluarch.detect_spindles(raw,sf,method='auto')
	assert adaptive['count']==multimethod['count']==6
	assert adaptive['parameters']['method_requested']=='fixed'
	assert multimethod['parameters']['method_requested']=='auto'


def test_spindle_dispatch_rejects_unknown_method() -> None:
	with pytest.raises(ValueError,match='method must be one of'):
		sluarch.detect_spindles(_raw(np.zeros(2_000)),method='invalid')
