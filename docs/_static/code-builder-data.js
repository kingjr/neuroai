/* AUTO-GENERATED from docs/_data/code-builder-data.yaml. DO NOT EDIT. */
window.CB_DATA = {
  "default_study": {
    "name": "FakeMulti",
    "comment": "Built-in synthetic multi-modal study (~2 MB) - covers any combination."
  },
  "studies": {
    "fmri-image": {
      "name": "Allen2022MassiveSample",
      "pip_packages": [
        "neuralfetch",
        "awscli"
      ],
      "comment": "Natural Scenes Dataset sample (Allen et al., 2022) - fMRI <-> natural images."
    },
    "fmri-audio": {
      "name": "Li2022PetitSample",
      "pip_packages": [
        "neuralfetch",
        "openneuro-py",
        "praatio"
      ],
      "comment": "Little Prince fMRI sample (Li et al., 2022) - fMRI <-> spoken story (audio + word + text)."
    },
    "fmri-text": {
      "name": "Li2022PetitSample",
      "pip_packages": [
        "neuralfetch",
        "openneuro-py",
        "praatio"
      ],
      "stim_kwargs": [
        "language=\"english\""
      ],
      "comment": "Little Prince fMRI sample (Li et al., 2022) - fMRI <-> word/text events from the audiobook."
    },
    "meg-audio": {
      "name": "Bel2026PetitListenSample",
      "pip_packages": [
        "neuralfetch",
        "openneuro-py"
      ],
      "neuro_kwargs": [
        "allow_maxshield=True"
      ],
      "comment": "Petit Prince MEG listening sample (Bel et al., 2026) - MEG <-> spoken story."
    },
    "meg-text": {
      "name": "Bel2026PetitListenSample",
      "pip_packages": [
        "neuralfetch",
        "openneuro-py"
      ],
      "neuro_kwargs": [
        "allow_maxshield=True"
      ],
      "stim_kwargs": [
        "language=\"french\""
      ],
      "comment": "Petit Prince MEG listening sample (Bel et al., 2026) - MEG <-> word/text events from the audiobook."
    },
    "eeg-image": {
      "name": "Grootswagers2022HumanSample",
      "pip_packages": [
        "neuralfetch",
        "pyunpack",
        "boto3",
        "osfclient",
        "openneuro-py"
      ],
      "comment": "THINGS-EEG sample (Grootswagers et al., 2022) - EEG <-> natural images."
    }
  },
  "axes": {
    "neuro": {
      "label": "Neuro",
      "default": "fmri",
      "options": {
        "meg": {
          "label": "MEG",
          "cls": "MegExtractor",
          "event_type": "Meg",
          "kwargs": [
            "frequency=120",
            "filter=(0.5, 30)",
            "scaler='RobustScaler'",
            "baseline=(-0.1, 0.0)"
          ],
          "yaml_extra": [
            "notch_filter=60.0",
            "apply_hilbert=False",
            "drop_bads=True",
            "mne_cpus=4"
          ],
          "window": {
            "start": -0.1,
            "duration": 0.5
          },
          "t": 48,
          "t_comment": "300ms @120Hz"
        },
        "eeg": {
          "label": "EEG",
          "cls": "EegExtractor",
          "event_type": "Eeg",
          "kwargs": [
            "frequency=120",
            "filter=(0.5, 40)",
            "scaler='RobustScaler'",
            "baseline=(-0.1, 0.0)"
          ],
          "yaml_extra": [
            "notch_filter=60.0",
            "apply_hilbert=False",
            "drop_bads=True",
            "mne_cpus=4"
          ],
          "window": {
            "start": -0.1,
            "duration": 0.5
          },
          "t": 48,
          "t_comment": "300ms @120Hz"
        },
        "ieeg": {
          "label": "iEEG",
          "cls": "IeegExtractor",
          "event_type": "Ieeg",
          "kwargs": [
            "frequency=100",
            "filter=(0.5, 40)",
            "picks=(\"seeg\",)"
          ],
          "yaml_extra": [
            "notch_filter=60.0",
            "drop_bads=True",
            "mne_cpus=4"
          ],
          "window": {
            "start": -0.1,
            "duration": 0.5
          },
          "t": 40,
          "t_comment": "300ms @100Hz"
        },
        "emg": {
          "label": "EMG",
          "cls": "EmgExtractor",
          "event_type": "Emg",
          "kwargs": [
            "frequency=256",
            "scaler='RobustScaler'"
          ],
          "yaml_extra": [
            "drop_bads=True",
            "mne_cpus=4"
          ],
          "window": {
            "start": -0.1,
            "duration": 0.5
          },
          "t": 102,
          "t_comment": "300ms @256Hz"
        },
        "fmri": {
          "label": "fMRI",
          "cls": "FmriExtractor",
          "event_type": "Fmri",
          "pip": [
            "tutorials"
          ],
          "kwargs": [
            "offset=5"
          ],
          "yaml_extra": [
            "fwhm=4.0",
            "allow_missing=False"
          ],
          "window": {
            "start": 0.0,
            "duration": 2.0
          },
          "x_expr": "batch.data[\"neuro\"].reshape(len(batch), -1)"
        },
        "fnirs": {
          "label": "fNIRS",
          "cls": "FnirsExtractor",
          "event_type": "Fnirs",
          "pip_packages": [
            "mne-nirs"
          ],
          "kwargs": [
            "frequency=10",
            "picks=(\"fnirs\",)",
            "scaler='RobustScaler'",
            "baseline=(-0.1, 0.0)"
          ],
          "yaml_extra": [
            "drop_bads=True",
            "mne_cpus=4",
            "compute_optical_density=False"
          ],
          "window": {
            "start": -0.1,
            "duration": 0.5
          },
          "t": 4,
          "t_comment": "300ms @10Hz"
        },
        "spike": {
          "label": "Spike",
          "cls": "SpikesExtractor",
          "event_type": "Spikes",
          "pip_packages": [
            "h5py"
          ],
          "kwargs": [
            "frequency=200",
            "baseline=(-0.1, 0.0)",
            "scaler='RobustScaler'"
          ],
          "yaml_extra": [
            "clamp=10.0",
            "scale_factor=1.0"
          ],
          "window": {
            "start": -0.1,
            "duration": 0.5
          },
          "t": 60,
          "t_comment": "200ms @200Hz"
        }
      }
    },
    "stim": {
      "label": "Stim",
      "default": "image",
      "options": {
        "text": {
          "label": "Text",
          "cls": "SpacyEmbedding",
          "pip": [
            "tutorials"
          ],
          "post_install": "python -m spacy download en_core_web_md",
          "event_type": "Word",
          "is_classification": false,
          "kwargs": [
            "aggregation=\"trigger\""
          ],
          "yaml_extra": []
        },
        "image": {
          "label": "Image",
          "cls": "HuggingFaceImage",
          "pip_packages": [
            "transformers",
            "torchvision"
          ],
          "event_type": "Image",
          "is_classification": false,
          "kwargs": [
            "model_name=\"facebook/dinov2-small\"",
            "aggregation=\"trigger\""
          ],
          "yaml_extra": [
            "imsize=224",
            "batch_size=16",
            "layer_aggregation=\"mean\"",
            "token_aggregation=\"mean\""
          ]
        },
        "audio": {
          "label": "Audio",
          "cls": "MelSpectrum",
          "event_type": "Audio",
          "pip_packages": [
            "julius",
            "torchaudio",
            "soundfile"
          ],
          "is_classification": false,
          "kwargs": [
            "n_mels=40",
            "aggregation=\"trigger\""
          ],
          "yaml_extra": [
            "n_fft=1024",
            "hop_length=256",
            "use_log_scale=True",
            "norm_audio=True"
          ]
        },
        "video": {
          "label": "Video",
          "cls": "HuggingFaceVideo",
          "pip_packages": [
            "transformers",
            "torchvision",
            "julius",
            "moviepy"
          ],
          "event_type": "Video",
          "is_classification": false,
          "l3_skip": "videomae+transformers ImagesKwargs(text=) compat \u2014 upstream",
          "kwargs": [
            "frequency=\"native\"",
            "use_audio=False",
            "aggregation=\"trigger\""
          ],
          "yaml_extra": [
            "num_frames=8",
            "max_imsize=224"
          ]
        },
        "classification": {
          "label": "Class",
          "cls": "LabelEncoder",
          "event_type": "Stimulus",
          "is_classification": true,
          "accepts_infra": false,
          "kwargs": [
            "event_types=\"Stimulus\"",
            "event_field=\"description\"",
            "return_one_hot=True",
            "aggregation=\"first\""
          ],
          "yaml_extra": [
            "treat_missing_as_separate_class=False"
          ]
        }
      }
    },
    "task": {
      "label": "Task",
      "default": "load",
      "options": {
        "load": {
          "label": "Load only",
          "needs_ml": false
        },
        "decoding": {
          "label": "Decoding",
          "needs_ml": true,
          "direction": "decoding"
        },
        "encoding": {
          "label": "Encoding",
          "needs_ml": true,
          "direction": "encoding"
        }
      }
    },
    "model": {
      "label": "Model",
      "default": "ridge",
      "options": {
        "ridge": {
          "label": "Ridge",
          "kind": "ridge"
        },
        "sgd": {
          "label": "SGD",
          "kind": "sgd"
        },
        "torch": {
          "label": "Torch",
          "kind": "torch"
        }
      }
    },
    "style": {
      "label": "Config",
      "default": "kwargs",
      "options": {
        "kwargs": {
          "label": "kwargs"
        },
        "yaml": {
          "label": "YAML"
        }
      }
    },
    "compute": {
      "label": "Compute",
      "default": "local",
      "options": {
        "local": {
          "label": "Local",
          "infra_literal": "{\"folder\": \"$CACHE\"}"
        },
        "slurm": {
          "label": "Slurm",
          "infra_literal": "{\"folder\": \"$CACHE\", \"cluster\": \"slurm\", \"gpus_per_node\": 1, \"slurm_partition\": \"gpu\", \"timeout_min\": 120}"
        }
      }
    }
  }
};
