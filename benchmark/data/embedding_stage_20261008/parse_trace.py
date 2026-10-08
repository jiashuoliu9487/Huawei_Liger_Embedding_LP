import collections
import json
import statistics
import sys

from pathlib import Path

p = Path(sys.argv[1])
r = json.loads(p.read_text())
events = r.get("traceEvents", []) if isinstance(r, dict) else r
values = collections.defaultdict(list)
for e in events:
    if (
        e.get("ph") == "X"
        and isinstance(e.get("dur"), (int, float))
        and any(t in e.get("name", "").lower() for t in ["embedding", "gather", "index_select", "triton"])
    ):
        values[(e.get("cat", ""), e["name"])].append(e["dur"])
for (cat, name), v in values.items():
    print(cat, name[:160], len(v), "median_us", statistics.median(v), "mean_us", statistics.mean(v))
