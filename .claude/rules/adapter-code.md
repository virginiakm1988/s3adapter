---
paths:
  - "s3prl/s3prl/run_downstream.py"
  - "s3prl/s3prl/downstream/**"
  - "s3prl/s3prl/upstream/wav2vec2/**"
  - "s3prl/s3prl/upstream/interfaces.py"
---
# Rules for editing the adapter modification layer

You are editing the live code of this project. Before your first edit, read
`.claude/docs/PROJECT-MAP.md` if you haven't this session.

## Hard rules (violating these breaks the user's experiments)

1. **Do not refactor the `sys.argv[-1]` mechanism.** Adapter selection works by
   substring match on the experiment name (`-n`, passed as the last CLI arg).
   It looks like a bug. It is the intended interface (README documents it).
   Do not replace it with `args.adapter`, do not reorder the if/elif chain in
   `wav2vec2_model.py:3282-3296` (AdapterBias's branch relies on being checked
   with `not houlsby / not lora` guards), and do not rename the `-n` convention.
   If the user explicitly asks to refactor this, see "escalation" in
   `.claude/docs/20-JUDGMENT.md` — it touches saved-checkpoint compatibility.

2. **Do not "fix" `--adapter` default `"None"` (string).** `if self.args.adapter:`
   being always-true is baked into runner.py's load/save paths. Changing the
   default to real `None` changes checkpoint contents. Only with user approval.

3. **New adapter parameters MUST contain `"adapter"` or `"lora"` in their
   attribute path** (e.g. `self.adapter = ...` so param names are
   `...adapter.weight`). Otherwise `runner.py:273-285` never sets
   `requires_grad=True` and the adapter silently doesn't train, and
   `runner.py:447-461` never saves it.

4. **A new adapter type needs edits in ALL of:** construction chain
   (`wav2vec2_model.py` ~3282), BOTH forward injection points (~3351 and ~3387,
   for the two layer-norm orderings), and the README table. Missing one =
   silent no-op for half the configurations.

5. **Do not delete the debug `print(...)` calls** (e.g. `Adapter!!`,
   conv shape prints). The user reads them to confirm which params train.
   They are the de facto test suite.

6. **Config YAMLs under `s3prl/s3prl/downstream/*/config.yaml` are experiment
   state.** Don't change values (lr, steps, batch size) as a side effect of
   another task. Confirm with the user first unless the task IS "change config".

## After every edit
- Run `python3 -m py_compile <file>` (torch isn't installed; imports beyond
  stdlib will fail at run time, so py_compile is the strongest local check).
- Run the L2 invariant checks in `.claude/docs/00-DIAGNOSIS.md` §3 when you
  touched adapter wiring.
- Report which verification level (L1/L2/L3) you reached. Never say "tested".
