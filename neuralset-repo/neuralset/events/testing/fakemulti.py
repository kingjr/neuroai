# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.

"""Multi-modal synthetic study used by the docs Code Builder page.

Generates a single shared dataset on disk that exposes events for every
neuro modality (MEG / EEG / iEEG / EMG / fMRI) and every stimulus type
(Word / Image / Audio / classification ``Stimulus``) the Code Builder
exposes — so any (neuro × stim) combination yields a runnable script.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

_DURATION_S = 60.0
_N_STIM = 20
_WORDS = (
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
    "runs",
    "fast",
    "today",
    "across",
    "the",
    "open",
    "field",
    "before",
    "the",
    "sun",
    "sets",
)


def _make_meg_like(folder, fname, ch_type, n_chans, sfreq, rng):
    """Write a tiny .fif with the requested channel type, only if missing."""
    import mne

    fp = folder / fname
    if fp.exists():
        return
    info = mne.create_info(
        ch_names=[f"{ch_type.upper()}{i:03d}" for i in range(n_chans)],
        sfreq=sfreq,
        ch_types=ch_type,
    )
    n_samples = int(sfreq * _DURATION_S)
    data = rng.randn(n_chans, n_samples).astype("float32") * 1e-6
    mne.io.RawArray(data, info, verbose="ERROR").save(
        fp,
        overwrite=True,
        verbose="ERROR",
    )


def _make_fmri(folder, fname, rng):
    import nibabel as nib

    fp = folder / fname
    if fp.exists():
        return
    n_vox, n_t = 12, int(_DURATION_S / 2.0)  # TR = 2s
    img = nib.Nifti1Image(
        rng.randn(n_vox, n_vox, n_vox, n_t).astype("float32"),
        np.eye(4),
    )
    img.header.set_zooms((3.0, 3.0, 3.0, 2.0))
    img.to_filename(fp)


def _make_images(folder, n, rng):
    from PIL import Image

    for i in range(n):
        fp = folder / f"img_{i:03d}.png"
        if fp.exists():
            continue
        arr = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(fp)


def _make_audio(folder, n, rng):
    import soundfile as sf

    for i in range(n):
        fp = folder / f"aud_{i:03d}.wav"
        if fp.exists():
            continue
        sf.write(fp, (rng.randn(8_000).astype("float32") * 0.1), 16_000)


def _make_spikes(folder, fname, rng, n_units=8, rate_hz=5.0):
    """Write a tiny NWB-style HDF5 file with spike trains."""
    import h5py

    fp = folder / fname
    if fp.exists():
        return
    spike_times: list[float] = []
    spike_times_index: list[int] = []
    cumulative = 0
    for _ in range(n_units):
        n = max(1, int(rng.poisson(rate_hz * _DURATION_S)))
        ts = np.sort(rng.uniform(0.0, _DURATION_S, size=n)).astype("float64")
        spike_times.extend(ts.tolist())
        cumulative += n
        spike_times_index.append(cumulative)
    with h5py.File(fp, "w") as f:
        units = f.create_group("units")
        units.create_dataset("id", data=np.arange(n_units, dtype="int32"))
        units.create_dataset("spike_times", data=np.asarray(spike_times))
        units.create_dataset(
            "spike_times_index",
            data=np.asarray(spike_times_index, dtype="int64"),
        )


class FakeMulti(study.Study):
    """Multi-modal synthetic study (2 subjects × 2 runs).

    Each timeline carries one minute of synthetic recordings for **every**
    neuro modality the Code Builder supports (MEG / EEG / iEEG / EMG / fMRI),
    plus 20 image, word, audio and ``Stimulus`` events. Used by
    ``docs/neuralset/code_builder.rst`` so any (neuro × stim) combination
    has data to chew on without an external download.
    """

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=4,
        num_subjects=2,
        num_events_in_query=4 * (6 + 4 * _N_STIM),
        event_types_in_query={
            "Meg",
            "Eeg",
            "Ieeg",
            "Emg",
            "Fmri",
            "Spikes",
            "Image",
            "Word",
            "Audio",
            "Stimulus",
        },
        data_shape=(0,),
        frequency=100.0,
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.infra_timelines.cluster = None  # purely sequential

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for s in (1, 2):
            for r in (1, 2):
                yield {"subject": f"sub-{s:02d}", "run": r}

    def _download(self) -> None:
        for tl in self.iter_timelines():
            self._materialize(tl)

    def _materialize(self, timeline: dict[str, tp.Any]):
        folder = self.path / timeline["subject"] / f"run-{timeline['run']}"
        folder.mkdir(parents=True, exist_ok=True)
        seed = (int(timeline["subject"][-2:]) - 1) * 100 + timeline["run"]
        rng = np.random.RandomState(seed)
        _make_meg_like(folder, "raw-meg.fif", "mag", 19, 120.0, rng)
        _make_meg_like(folder, "raw-eeg.fif", "eeg", 32, 256.0, rng)
        _make_meg_like(folder, "raw-ieeg.fif", "seeg", 16, 512.0, rng)
        _make_meg_like(folder, "raw-emg.fif", "emg", 8, 256.0, rng)
        _make_fmri(folder, "bold.nii.gz", rng)
        _make_spikes(folder, "spikes.h5", rng)
        _make_images(folder, _N_STIM, rng)
        _make_audio(folder, _N_STIM, rng)
        return folder

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        folder = self._materialize(timeline)
        rows: list[dict[str, tp.Any]] = []
        # neuro recordings
        for typ, fname in (
            ("Meg", "raw-meg.fif"),
            ("Eeg", "raw-eeg.fif"),
            ("Ieeg", "raw-ieeg.fif"),
            ("Emg", "raw-emg.fif"),
        ):
            rows.append(
                {
                    "type": typ,
                    "filepath": str(folder / fname),
                    "start": 0.0,
                    "duration": _DURATION_S,
                }
            )
        rows.append(
            {
                "type": "Fmri",
                "filepath": str(folder / "bold.nii.gz"),
                "start": 0.0,
                "duration": _DURATION_S,
                "frequency": 0.5,
            }
        )
        rows.append(
            {
                "type": "Spikes",
                "filepath": str(folder / "spikes.h5"),
                "start": 0.0,
                "duration": _DURATION_S,
                "frequency": 200.0,
                "subject": str(timeline["subject"]),
            }
        )
        # stim events (uniformly spread across the recording)
        rng = np.random.RandomState(int(timeline["run"]))
        classes = ("face", "object", "scene")
        for i in range(_N_STIM):
            t = 1.0 + i * 1.0
            rows.append(
                {
                    "type": "Word",
                    "start": t,
                    "duration": 0.3,
                    "text": _WORDS[i % len(_WORDS)],
                    "language": "english",
                }
            )
            rows.append(
                {
                    "type": "Image",
                    "start": t,
                    "duration": 0.3,
                    "filepath": str(folder / f"img_{i:03d}.png"),
                }
            )
            rows.append(
                {
                    "type": "Audio",
                    "start": t,
                    "duration": 0.5,
                    "filepath": str(folder / f"aud_{i:03d}.wav"),
                }
            )
            rows.append(
                {
                    "type": "Stimulus",
                    "start": t,
                    "duration": 0.3,
                    "description": str(rng.choice(classes)),
                }
            )
        return pd.DataFrame(rows)
