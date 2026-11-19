# ELMA (formerly HiSAV)

**Explicit Localization and Memory for Audiovisual separation**

Implementation accompanying *When Short Visual Context Is Ambiguous: Explicit Localization and Temporal Memory for Low-Latency Speech Separation*.

Short windows can contain little lip motion, making the visual evidence ambiguous. A two-output separator can also return the same speaker at a different output in the next window. ELMA combines explicit target localization with temporal memory to select the visible speaker’s voice from six-channel mixtures. The method is now named **ELMA**; this repository retains the historical HiSAV name and the `hisav` Python package. The corresponding [ELMA repository](https://github.com/vu-duy-tung/ELMA) contains the same implementation code.

## Method

![ELMA pipeline: array-matched SAVE synthesis supplies direction labels; visually conditioned IPDNet2 learns a spatial-audiovisual representation; candidate alignment and temporal memory guide target-stream selection.](https://raw.githubusercontent.com/vu-duy-tung/ELMA/a89f6477124c0d7d4e27872f712fbe93ca41428d/docs/figures/elma-pipeline.png)

1. **Array-matched synthesis.** SAVE simulates Mandarin mixtures for a specified microphone array under varied acoustic and visual conditions. Target directions supervise localization, and clean target waveforms supervise stream selection.
2. **Explicit target localization.** Visually conditioned IPDNet2 learns to associate the target mouth video with a source in the multichannel mixture. The selector uses the resulting spatial-audiovisual representation to complement short-window visual evidence.
3. **Temporal memory.** Candidate alignment connects separator outputs across windows. Candidate-specific states and speaker identity cues use earlier evidence to refine the audiovisual selection scores.

The released implementation uses waveform-overlap correlation for candidate alignment and speaker embeddings for score correction; see the [recipe and implementation details](docs/recipe.md). With 1 s windows, the system uses 0.4 s of past context, emits 0.4 s, and uses 0.2 s of lookahead. The initial input wait is therefore **0.6 s before computation**, rather than 1 s or the processing time alone.

## Results reported in the paper

Character error rate (CER; lower is better) on the **2,000-scene SAVE test set**, with both speakers evaluated as targets. Each listed system has one or more components fine-tuned on SAVE.

| Method | 1 s CER | 4 s CER |
| --- | ---: | ---: |
| DAVE + TalkNet | 41.50% | 23.61% |
| AV-TFGridNet | 36.68% | 21.95% |
| AV-MossFormer2 | 36.50% | 26.71% |
| **ELMA** | **22.54%** | **18.62%** |

ELMA and the fine-tuned DAVE + TalkNet baseline share the same separator. With 1 s windows, ELMA takes approximately **152 ms per window on an NVIDIA H20**; this processing time excludes the input wait. On the campus recordings, SAVE-trained ELMA obtains **7.85% CER**, compared with **11.52%** for DAVE + TalkNet.

These are the paper’s evaluated results. The SAVE and campus evaluations use distinct checkpoint versions; the locally supplied bundle’s `PROVENANCE.json` identifies which version is being run. The consolidated training command is not a claim of bitwise reproduction of historical training.

## Run the code

Follow [docs/setup.md](docs/setup.md) to install the dependencies, prepare a compatible model bundle in `assets/default`, and set `SAVE` and `OUTPUT` to local directories. The SAVE paper split contains **32,768 training, 1,000 validation and 2,000 test scenes**. Model weights and dataset files are external assets and are not included in this repository.

**Validate the assets without initializing a model or GPU:**

```bash
python -m hisav train --data "$SAVE" --bundle assets/default --output "$OUTPUT/training" --check
```

**Fine-tune the components and selector on all 32,768 training scenes:**

```bash
python -m hisav train --data "$SAVE" --bundle assets/default --output "$OUTPUT/training"
```

**Infer on the 2,000 test scenes using the supplied bundle:**

```bash
python -m hisav infer --data "$SAVE" --bundle assets/default --output "$OUTPUT/evaluation"
```

See [docs/recipe.md](docs/recipe.md) for training stages, supervision and checkpoint provenance, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the vendored components.
