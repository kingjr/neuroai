from pathlib import Path
import neuralset as ns
from torch.utils.data import DataLoader

CACHE = Path.home() / "neuroai_data" / ".cache"
STUDIES = Path.home() / "neuroai_data" / "studies"
STUDIES.mkdir(parents=True, exist_ok=True)
infra = {"folder": CACHE}

# 1. Built-in synthetic multi-modal study (~2 MB) - covers any combination.
study = ns.Study(
    name="FakeMulti",
    path=STUDIES,
    infra_timelines={"folder": CACHE},
)

# 2. Define extractors
neuro = ns.extractors.SpikesExtractor(
    frequency=200,
    baseline=(-0.1, 0.0),
    scaler='RobustScaler',
    infra=infra,
)
stim  = ns.extractors.HuggingFaceImage(
    model_name="facebook/dinov2-small",
    aggregation="trigger",
    infra=infra,
)

# 3. Segment around each "Image" event
segmenter = ns.Segmenter(
    start=-0.1, duration=0.5,
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
