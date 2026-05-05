from pathlib import Path
import neuralset as ns
from torch.utils.data import DataLoader

CACHE = Path.home() / "neuroai_data" / ".cache"
STUDIES = Path.home() / "neuroai_data" / "studies"
STUDIES.mkdir(parents=True, exist_ok=True)
infra = {
    "folder": CACHE,
    "cluster": "slurm",
    "gpus_per_node": 1,
    "slurm_partition": "gpu",
    "timeout_min": 120,
}

# 1. Natural Scenes Dataset sample (Allen et al., 2022) - fMRI <-> natural images.
study = ns.Study(
    name="Allen2022MassiveSample",
    path=STUDIES,
    infra={"folder": CACHE},
)

# 2. Define extractors
neuro = ns.extractors.FmriExtractor(
    offset=5,
    infra=infra,
)
stim  = ns.extractors.HuggingFaceImage(
    model_name="facebook/dinov2-small",
    aggregation="trigger",
    infra=infra,
)

# 3. Segment around each "Image" event
segmenter = ns.Segmenter(
    start=0, duration=2,
    trigger_query='type=="Image"',
    extractors=dict(neuro=neuro, stim=stim),
    drop_incomplete=True,
)

# 4. Run the study and apply the segmenter
study.download()
events = study.run()  # simple pd.DataFrame
dset = segmenter.apply(events)
dset.prepare()

# 5. Inspect one batch of 8 segments
loader = DataLoader(dset, batch_size=8, collate_fn=dset.collate_fn)
batch = next(iter(loader))
print('neuro', batch.data["neuro"].shape)
print('stim', batch.data["stim"].shape)
