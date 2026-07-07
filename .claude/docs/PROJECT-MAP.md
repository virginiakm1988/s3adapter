# Project map — where the live code is (check here BEFORE searching)

s3adapter = a fork of the s3prl speech-SSL toolkit that adds parameter-efficient
fine-tuning (adapters) to upstream models, plus a vendored fairseq. The fork's
entire value lives in ~5 files; everything else is upstream code.

Line anchors below were verified 2026-07-07. Code shifts over time: treat
anchors as "start grepping here", and re-verify with `Grep -n` before editing.
If an anchor is >30 lines off, fix it here (see `40-MAINTENANCE.md`).

## The live files (the actual project)

### 1. `s3prl/s3prl/run_downstream.py` (237 lines) — CLI entry point
- `:96` — `--adapter` flag. **Default is the STRING `"None"`, not None.**
- `:99` — `--prompt` flag (prefix/preinput prompt tuning, a sibling feature).
- The experiment name `-n <name>` becomes `sys.argv[-1]` when passed last —
  and adapter injection keys off substrings of it (see file 3).

### 2. `s3prl/s3prl/downstream/runner.py` (648 lines) — training loop
- `:110-125` — loads saved prompt/adapter weights from a previous experiment
  (`self.init_ckpt.get("adapter")`). `if self.args.adapter:` is ALWAYS true
  (string `"None"` is truthy).
- `:273-285` — marks params trainable: any param whose name contains
  `"adapter"` or `"lora"` gets `requires_grad = True`; prints `Adapter!!<name>`.
  Everything else in the Upstream stays frozen. **Invariant: new adapter
  modules MUST have "adapter" or "lora" in their parameter names**, or they
  silently won't train.
- `:447-461` — checkpoint saving: collects adapter-named params into
  `all_states["adapter"]`; `--adapter bitfit` additionally saves all `bias` params.
- Upstream param removed from optimizer when using adapters (commit 31a54ef).

### 3. `s3prl/s3prl/upstream/wav2vec2/wav2vec2_model.py` (3,424 lines) — the model fork
Vendored fairseq wav2vec2/HuBERT model with adapters spliced in. NEVER read whole.
- `:12` — `import loralib as lora`.
- `:846-853` — LoRA: `if 'lora' in sys.argv[-1]:` replaces the k_proj and
  q_proj of `MultiheadAttention` (class at `:778`) with `lora.Linear(..., r=8)`.
- `:2874` — `ConvFeatureExtractionModel`; `:2970-2980` — CNN adapter in the
  feature extractor: `x = conv(x) + 0.01*cnn_adapter(x)` (note the debug
  `print(...shape)` — intentional, do not "clean up" without asking).
- `:3249` — `TransformerSentenceEncoderLayer`:
  - `:3282-3296` — adapter module construction, selected by substring of
    `sys.argv[-1]`: order is AdapterBias (`'adapter' in name` and NOT
    houlsby/lora) → Houlsby (`'houlsby' in name`) → ... Order is load-bearing.
  - `:3351-3396` — forward-pass injection (two insertion points, both
    layer-norm orderings). Houlsby uses `houlsby_input` captured pre-block.
- HuBERT weights load via this file, not via `fairseq/`.

### 4. `s3prl/s3prl/upstream/wav2vec2/expert.py` (173 lines) — checkpoint loader
- `:78, :108, :144` — model loading via `fairseq.checkpoint_utils`. Loading is
  deliberately tolerant (README: strict=False) so adapter-augmented models can
  load vanilla checkpoints. Side effect: key mismatches fail SILENTLY.

### 5. `s3prl/s3prl/upstream/interfaces.py`
- `:217` — assumption removed to allow weighted-sum during fine-tuning/adapters.

## Secondary (edit with care)
- `s3prl/s3prl/downstream/<task>/config.yaml` — per-task experiment configs
  (asr, speech_commands, fluent_commands, ...). Changing these changes the
  user's experiment results; confirm before editing.
- `s3prl/s3prl/downstream/model.py`, `specaug.py` — touched by the reverted
  multitask work (d2ebb64 / 48f8082); currently back to pre-multitask state.

## Vendored / read-only (do not edit, do not search by default)
- `fairseq/` — vendored fairseq, incl. committed `.eggs/` and `build.zip`.
  Only relevant when debugging checkpoint loading via `fairseq.checkpoint_utils`.
- `s3prl/s3prl/upstream/*` other than `wav2vec2/` and `interfaces.py` —
  ~20 untouched upstream model families.
- `s3prl/docs/`, `s3prl/test/`, root `*.jpg` — irrelevant to all tasks.

## Known supported adapters (README + code)
| `-n` name contains | Adapter | Where built |
|---|---|---|
| `adapter` (w/o houlsby/lora) | AdapterBias | wav2vec2_model.py:3282 |
| `houlsby` | Houlsby bottleneck | :3286 |
| `lora` | LoRA r=8 on k/q proj | :846-853, gated on `'lora' in sys.argv[-1]` |
| `bitfit` | bias-only tuning | runner.py (param freezing) |
| `cnn` | CNN adapter in feature extractor | :2970-2980 |
Combinations work by substring: `-n hubert_asr_houlsby_cnn` = Houlsby + CNN.
