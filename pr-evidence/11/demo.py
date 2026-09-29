import io, statistics, subprocess, tempfile, time, zipfile
import fsspec, pandas as pd
from fsspec.implementations.memory import MemoryFileSystem
from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage

rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                     cwd=ParquetReaderStage.__module__ and __import__("nemo_curator").__path__[0]).stdout.strip()
print(f"ParquetReaderStage.read_data @ {rev}   pandas {pd.__version__}")

class SlowMem(MemoryFileSystem):  # stands in for object-store latency: 20 ms per read()
    protocol = ("slowmem",)
    def _open(self, path, mode="rb", **kw):
        f = super()._open(path, mode=mode, **kw); r = f.read
        def slow(*a, **k): time.sleep(0.02); return r(*a, **k)
        f.read = slow; return f
fsspec.register_implementation("slowmem", SlowMem, clobber=True)

def ref(paths):
    return pd.concat((pd.read_parquet(p, engine="pyarrow", dtype_backend="pyarrow") for p in paths), ignore_index=True)

d = tempfile.mkdtemp()
paths = []
for i in range(50):
    p = f"{d}/part_{i}.parquet"
    pd.DataFrame({"id": range(i * 20000, (i + 1) * 20000), "text": ["x" * 20] * 20000}).to_parquet(p, index=False)
    paths.append(p)
    with open(p, "rb") as src, fsspec.open(f"slowmem://g/part_{i}.parquet", "wb") as dst:
        dst.write(src.read())
stage = ParquetReaderStage(read_kwargs={"storage_options": {}})
for label, ps in [("local, 50 x 20,000 rows", paths), ("20 ms/read, 50 x 20,000 rows", [f"slowmem://g/part_{i}.parquet" for i in range(50)])]:
    times = []
    for _ in range(3):
        t = time.perf_counter(); df = stage.read_data(ps); times.append(time.perf_counter() - t)
    print(f"  {label:30s} {statistics.median(times):6.3f} s   rows={len(df):,}  equals per-file concat: {df.equals(ref(paths))}")

def check(label, ps):
    try:
        got = ParquetReaderStage().read_data(ps)
        print(f"  {label:30s} columns={got.columns.tolist()}  equals per-file concat: {got.equals(ref(ps))}")
    except Exception as e:
        print(f"  {label:30s} raised {type(e).__name__}: {str(e)[:60]}")

ps = [f"{d}/r.parquet", f"{d}/n.parquet"]
pd.DataFrame({"a": [1]}).to_parquet(ps[0]); pd.DataFrame({"a": [2]}, index=pd.Index([10], name="idx")).to_parquet(ps[1])
check("RangeIndex file + named index", ps)
zps = []
for name, v in [("a", 1), ("b", 2)]:
    with fsspec.open(f"memory://z/{name}.zip", "wb") as f, zipfile.ZipFile(f, "w") as z:
        z.writestr("part.parquet", pd.DataFrame({"a": [v]}).to_parquet(index=False))
    zps.append(f"zip://part.parquet::memory://z/{name}.zip")
check("zip:// in two archives", zps)
