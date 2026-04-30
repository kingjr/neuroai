from pathlib import Path
import neuralset as ns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

CACHE = Path.home() / "neuroai_data" / ".cache"
STUDIES = Path.home() / "neuroai_data" / "studies"
STUDIES.mkdir(parents=True, exist_ok=True)
infra = {"folder": CACHE}

# 1. Natural Scenes Dataset sample (Allen et al., 2022) - fMRI <-> natural images.
study = ns.Study(
    name="Allen2022MassiveSample",
    path=STUDIES,
    infra_timelines={"folder": CACHE},
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

# 5. Encoding (stim -> brain)
batch = dset.load_all()
scores = cross_val_score(
    estimator=Ridge(),
    X=batch.data["stim"].reshape(len(batch), -1),
    y=batch.data["neuro"].reshape(len(batch), -1),
)
print(f"score = {scores.mean():.3f}")
