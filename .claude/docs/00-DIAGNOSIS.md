# Harness Diagnosis — the 3 biggest failure points in this repo

Written 2026-07-07 by a Fable 5 session after direct inspection of the repo.
Every other file in `.claude/docs/` assumes you have read this one.
Each item below: what goes wrong, evidence, and the fix you must apply.

## 1. Token bleed: searching/reading 232 MB of vendored code that is not the project

**What goes wrong.** The repo tracks 3,971 files, but the actual project — the
adapter modification layer — is about 5 files (see `.claude/docs/PROJECT-MAP.md`).
`fairseq/` (124 MB, including committed `.eggs/` and `build.zip` build artifacts)
and most of `s3prl/` (108 MB) are vendored upstream code this fork barely touches.
A repo-wide `grep adapter` or reading `wav2vec2_model.py` (3,424 lines ≈ 35k tokens)
end-to-end burns a third of a context window and buries the signal.

**Fix (mandatory defaults):**
- Default search scope is: `s3prl/s3prl/run_downstream.py`,
  `s3prl/s3prl/downstream/`, `s3prl/s3prl/upstream/wav2vec2/`,
  `s3prl/s3prl/upstream/interfaces.py`.
- Search `fairseq/` ONLY if the task explicitly names fairseq. Never search
  `fairseq/.eggs/`, `fairseq/build*`, `s3prl/docs/` at all.
- Never read a file >500 lines end-to-end in the main conversation. Use
  `Grep -n` to find anchors, then `Read` with `offset`/`limit` (≤120 lines per read).
  Line anchors for the hot files are pre-computed in `PROJECT-MAP.md` — check
  there before searching at all.
- Any search you cannot scope in advance ("where is X handled?" with unknown
  location) goes to an `Explore` subagent, which returns only `file:line`
  conclusions (see `10-DISPATCH.md`).

## 2. Focus/error trap: adapter control flow is hidden in the experiment NAME, not the flags

**What goes wrong.** The code looks like `--adapter houlsby` selects the adapter.
It does not. Adapter injection into the upstream model is decided by **substring
matching on `sys.argv[-1]`** — the experiment name passed via `-n`:
- `s3prl/s3prl/upstream/wav2vec2/wav2vec2_model.py:3282-3296` (which adapter gets
  built) and `:3351-3396` (forward-pass injection): `'houlsby' in sys.argv[-1]`,
  `'lora' in sys.argv[-1]`, `'cnn' in sys.argv[-1]`, etc.
- Branch order matters: the AdapterBias branch is
  `'adapter' in sys.argv[-1] and 'houlsby' not in ... and 'lora' not in ...` —
  reordering the if-chain silently changes which adapter runs.
- Worse: `--adapter` defaults to the **string** `"None"`
  (`run_downstream.py:96`), so `if self.args.adapter:` and
  `self.args.adapter != None` in `runner.py` are ALWAYS true. The flag's literal
  value only matters where compared to `"bitfit"` (`runner.py:458`).

A model that doesn't know this will (a) add a new adapter and never see it
activate, (b) "fix" the truthiness bug and break checkpoint saving, or
(c) rename an experiment (`-n foo`) and unknowingly disable/enable adapters.

**Fix:** the full precedence table lives in `.claude/rules/adapter-code.md`
(auto-loads when you edit these files). Hard rule: never refactor the
`sys.argv[-1]` mechanism or the `"None"`-string default without explicit user
approval — experiments and saved checkpoints depend on current behavior.

## 3. Error/honesty trap: nothing here can actually run — false "verified" claims

**What goes wrong.** This container has no `torch`, no `fairseq` install, no GPU,
no dataset. Past failure modes: burning 50k+ tokens on `pip install torch` (it
won't make training runnable), or claiming a change is "tested" when at most it
was eyeballed. Extra hazard: upstream checkpoint loading is deliberately
non-strict (see `expert.py` / README note on `strict=False`), so wrong parameter
names fail SILENTLY at load time and only surface as garbage training metrics on
the user's machine.

**Fix — the verification ladder.** Always state which level you reached; never
imply a higher one:
1. **L1 — compiles:** `python3 -m py_compile <changed files>`. Minimum for any edit.
2. **L2 — invariants:** grep-based checks that the change preserves known
   invariants (adapter params contain `"adapter"`/`"lora"` in their name so
   `runner.py:273-280` marks them trainable; new branch added to BOTH the init
   chain ~3282 and forward chain ~3351/3387 of `wav2vec2_model.py`).
3. **L3 — user runs it:** hand the user an exact command
   (e.g. `python3 run_downstream.py --adapter houlsby -u hubert -d asr -m train -f -n hubert_asr_houlsby`)
   plus 2-3 specific things to check in the output (e.g. "the printed
   `Adapter!!` lines should list ~X params; loss should decrease in first 100 steps").
   Report as: "verified to L2; L3 requires your GPU box."

---
**Referenced by:** `CLAUDE.md` (routing), `10-DISPATCH.md` (what to delegate),
`20-JUDGMENT.md` (done-ness definitions), `40-MAINTENANCE.md` (updating this file).
