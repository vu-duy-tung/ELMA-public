# Environment and local assets

Use Python 3.11, a CUDA-enabled PyTorch build, and a compatible C++/CUDA compiler. From the repository root:

```bash
pip install -e .
pip install mamba-ssm==2.2.5 --no-build-isolation
```

The Mamba extension must match the installed PyTorch/CUDA versions. Run commands from the repository root; the editable installation resolves the vendored runtime there.

Place a compatible, separately supplied model bundle in `assets/default`, or pass its directory with `--bundle`. The bundle contains `selector.pt`, `spatial.pt`, `dave.pt`, `talknet.pt`, `voice.pt`, `config.json`, and `PROVENANCE.json`. The provenance file must contain a `files` mapping from relative filenames to SHA-256 digests (or objects with a `sha256` field). The command validates these hashes before loading a model. Model files are not downloaded automatically.

Set `$SAVE` to the local SAVE dataset directory, containing `manifests/tr.jsonl`, `manifests/cv.jsonl`, `manifests/tt.jsonl` and their referenced assets. The paper split has 32,768 training, 1,000 validation and 2,000 test scenes. Set `$OUTPUT` to a writable output directory, preferably on large-capacity storage. Dataset/model download URLs are omitted from this review snapshot.

Validate local assets and split counts without initializing a model or GPU:

```bash
python -m elma train --data "$SAVE" --bundle assets/default --output "$OUTPUT/training" --check
```

Inference supports disjoint `--rank` / `--world` scene shards, with each process assigned a device using `CUDA_VISIBLE_DEVICES`. The Python API reads `ELMA_BUNDLE` before importing model modules; the command-line interface sets it from `--bundle`.

Vendored component notices apply separately from pretrained weights and dataset assets. This snapshot does not establish redistribution rights for externally supplied assets.
