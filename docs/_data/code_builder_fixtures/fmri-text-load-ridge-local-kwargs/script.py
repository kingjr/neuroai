from pathlib import Path
import neuralset as ns
from torch.utils.data import DataLoader

CACHE = Path.home() / "neuroai_data" / ".cache"
STUDIES = Path.home() / "neuroai_data" / "studies"
STUDIES.mkdir(parents=True, exist_ok=True)
infra = {"folder": CACHE}

# 1. Little Prince fMRI sample (Li et al., 2022) - fMRI <-> word/text events from the audiobook.
study = ns.Study(
    name="Li2022PetitSample",
    path=STUDIES,
    infra_timelines={"folder": CACHE},
)

# 2. Define extractors
neuro = ns.extractors.FmriExtractor(
    offset=5,
    infra=infra,
)
stim  = ns.extractors.SpacyEmbedding(
    aggregation="trigger",
    language="english",
    infra=infra,
)

# 3. Segment around each "Word" event
segmenter = ns.Segmenter(
    start=0, duration=2,
    trigger_query='type=="Word"',
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
