# Mechanism Memory

Code and saved results for **Causal Retention in Interactive Agents**.
The current implementation is in [`software/`](software/), with experiment
drivers for finite interventions, continuous simulators, TD-MPC2 response
retention, and Qwen hidden-state probes.

## Current Version

Use the assets attached to release **v2026.11.21** on the
[Releases page](https://github.com/ShengjunZhang/mechanism-memory/releases):

- `code.zip`: algorithms, experiment drivers, analysis scripts and tests.
- `data.zip`: saved results, protocol records, file hashes and reproduction guide.
- `SHA256SUMS.txt`: checksums for both archives.

Extract `code.zip` and `data.zip` into the same empty directory. Run commands
from that directory using Python 3.12:

```text
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py validate
python reproduce.py tests
python reproduce.py figures --output ../reproduced_figures
```

The figure output directory must be new and outside the extracted project.
These commands check and analyze saved results. Fresh experiments are described
in `README.md` and `REPRODUCTION.txt` inside the extracted archives; TD-MPC2 and
Qwen runs require the specified model checkpoints.

## Working From a Clone

Enter `software/` and extract the release's `data.zip` **inside that directory**.
Then run the same commands there. The 117 files in `software/` match `code.zip`
byte for byte. Data and model weights are not committed to this directory.

The automatically generated GitHub source archives contain the entire
repository, including legacy files. Use the attached `code.zip` and `data.zip`
for the reproduction layout described above.

## Earlier Version

The earlier implementation, result files and scripts remain at the repository
root for reference. Do not mix them with `software/`. Their original entry
points are described in [`docs/legacy_README.md`](docs/legacy_README.md).
The historical snapshot cited in the paper remains commit
[`c1655d69`](https://github.com/ShengjunZhang/mechanism-memory/tree/c1655d69b5a8316b67d5f09cac0a4df9f97c8fe6).

The bundled TD-MPC2 source retains its upstream
[license](software/external/tdmpc2/LICENSE).
