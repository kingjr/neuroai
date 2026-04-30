from pathlib import Path
import yaml, pydantic
import neuralset as ns
from torch.utils.data import DataLoader

config = '''
# Natural Scenes Dataset sample (Allen et al., 2022) - fMRI <-> natural images.
study:
  name: Allen2022MassiveSample
  path: $STUDIES/Allen2022MassiveSample
  infra_timelines:
    folder: $CACHE
segmenter:
  start: 0
  duration: 2
  trigger_query: 'type=="Image"'
  drop_incomplete: true
  extractors:
    neuro:
      name: FmriExtractor
      offset: 5
      infra:
        folder: $CACHE
    stim:
      name: HuggingFaceImage
      model_name: facebook/dinov2-small
      aggregation: trigger
      infra:
        folder: $CACHE
'''

CACHE = Path.home() / "neuroai_data" / ".cache"
STUDIES = Path.home() / "neuroai_data" / "studies"
STUDIES.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load(
    config.replace("$CACHE", str(CACHE))
          .replace("$STUDIES", str(STUDIES))
)

class Experiment(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    study: ns.Study
    segmenter: ns.Segmenter

exp = Experiment(**config)
exp.study.download()
events = exp.study.run()  # simple pd.DataFrame
dset = exp.segmenter.apply(events)
dset.prepare()

# Inspect one batch of 8 segments
loader = DataLoader(dset, batch_size=8, collate_fn=dset.collate_fn)
batch = next(iter(loader))
print('neuro', batch.data["neuro"].shape)
print('stim', batch.data["stim"].shape)
