import pandas as pd

from nemo_curator.stages.text.filters.heuristic import BoilerPlateStringFilter
from nemo_curator.stages.text.modifiers import BoilerPlateStringModifier
from nemo_curator.tasks import DocumentBatch

DOC = "\n".join([
    "Home | News | Sport | Weather | Contact",
    "The garden club met on Sunday morning to plant roses along the path.",
    "Volunteers brought spades, gloves and forty seedlings from the local nursery.",
    "https://example.com/share?utm_source=twitter",
    "This website uses cookies, see our privacy policy.",
    "The club secretary thanked the council for donating fresh compost this year.",
    "Next month the group will prune the apple trees near the old gate.",
])
print("INPUT:\n" + DOC + "\n")

f = BoilerPlateStringFilter()
print(f"BoilerPlateStringFilter keep_document: {f.keep_document(f.score_document(DOC))}")
m = BoilerPlateStringModifier()
print(f"BoilerPlateStringModifier unchanged: {m.modify_document(DOC) == DOC}\n")

try:
    from nemo_curator.stages.text.filters import LineLevelQualityFilter
except ImportError as e:
    print(f"ImportError: {e}")
    raise SystemExit(1)

stage = LineLevelQualityFilter(nav_pattern=r"^\s*\w+(\s*\|\s*\w+){2,}\s*$")
out = stage.process(DocumentBatch(data=pd.DataFrame({"text": [DOC]}), dataset_name="demo"))
print("LineLevelQualityFilter output:\n" + out.to_pandas()["text"].iloc[0] + "\n")
print({k: int(v) for k, v in stage._consume_custom_metrics().items() if v})
