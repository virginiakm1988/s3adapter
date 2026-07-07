# Judgment rubrics — decisions made checkable

These rubrics replace "use good judgment" with tests you can run.
Each has a ✅ example (apply the rule) and a ❌ example (rule does not apply).

## 1. When to escalate the model (see 10-DISPATCH.md Rule 6 for mechanics)

Escalate when ANY of:
- The task requires holding >2 interacting constraints at once (e.g. checkpoint
  compatibility AND the sys.argv mechanism AND param-naming invariant).
- Two candidate answers both look plausible and you cannot construct a check
  that discriminates them.
- The fix you're about to make is in a different file than the symptom, and
  you can't state the causal chain in ≤3 steps.
- You are about to write "should work" / "probably" in a report.

✅ Escalate: "Loss doesn't decrease with houlsby adapter. I found the params ARE
marked trainable (runner.py:273 prints them). Two hypotheses: optimizer excludes
them (commit 31a54ef changed that code) or forward path never adds them. Both
look plausible." → interacting constraints + two live hypotheses → opus.
❌ Don't escalate: "grep found no `bitfit` branch in wav2vec2_model.py." Correct —
bitfit works by freezing in runner.py, not by injection. That's in
PROJECT-MAP.md; reading the map is cheaper than any escalation.

## 2. When something is truly DONE

Done = ALL of:
- [ ] Every acceptance condition from the dispatch/task is individually checked
      (not "overall looks good").
- [ ] Verification level explicitly stated: L1 py_compile / L2 invariants /
      L3 user-run (00-DIAGNOSIS.md §3). Code changes need ≥L2 to be called done.
- [ ] A FRESH agent (not you) confirmed the acceptance conditions (10-DISPATCH.md Rule 5).
- [ ] Work is committed AND pushed to the designated branch.
- [ ] The report says what was NOT verified.

✅ Done: "Added cnn2 adapter. L2: py_compile passes; branch present at :3282,
:3351, :3387; params named `cnn2_adapter.*` match runner's 'adapter' filter.
Read-back agent confirmed. Pushed. NOT verified: actual training (needs your GPU;
run <cmd> and check `Adapter!!` lines list cnn2_adapter params)."
❌ Not done: "Implemented the adapter and it compiles" — no L2, no fresh-agent
check, unpushed. That is 60% done; do not report it as done.

## 3. When to stop and ask the user

Ask (with AskUserQuestion when available, else state the blocker and end turn)
when ANY of:
- The action is irreversible or outward-facing: changing checkpoint format,
  config YAML values, deleting anything, force-push, opening a PR.
- Two verifiers/second opinions disagree after one clarification round.
- Two full retry rounds exhausted (dispatch Rule 6 hard cap).
- The task is ambiguous in a way that changes >30% of the work (e.g. "add
  adapter to hubert" — the hubert upstream dir, or the hubert checkpoint via
  the wav2vec2 path? Different scopes entirely).
- The request conflicts with a hard rule in `.claude/rules/`.

Format: 1-2 sentences of context, the exact decision needed, your recommendation.
✅ Ask: "Refactoring sys.argv selection to --adapter would break loading of all
existing experiment checkpoints (name-keyed). Proceed anyway, or add a
compatibility shim? I recommend the shim."
❌ Don't ask: "Which file should I edit?" when PROJECT-MAP.md answers it. Never
ask questions a listed file answers.

## 4. Wrong-direction signals — switch paths instead of retrying

If ANY of these fire, STOP the current approach. Do not retry a variation.
Go back to the last point where you had verified facts and pick a different route:
- Your fix requires editing vendored code (`fairseq/`, non-wav2vec2 upstreams).
  The real fix is almost always in the 5 live files.
- Each attempt grows the diff (fix #2 patches fix #1). Two generations of
  patches-on-patches = revert to clean state, re-diagnose.
- You're about to install packages (torch, fairseq) to "check something."
  Nothing runs here (CLAUDE.md fact 3); design an L2 grep-invariant instead.
- A search "finds nothing" for something that must exist → your scope or
  pattern is wrong (likely: it's keyed off `sys.argv[-1]`, or named
  differently, e.g. AdapterBias is just `adapter`). Recheck PROJECT-MAP.md
  before widening the search.
- You catch yourself explaining why the unexpected result is "actually fine."

✅ Switch: "To make LoRA work with `-d asr` I need to change
fairseq/fairseq/modules/multihead_attention.py" → wrong direction; the fork's
attention lives in wav2vec2_model.py:778.
❌ Don't switch: first py_compile failure on YOUR new code is a normal fix-it,
not a direction signal — that's within the two-retry budget.

## 5. Quality floor — verify before any hand-off

Minimum checks before reporting any code change (run all; they're cheap):
1. `python3 -m py_compile <every changed .py>` — must pass.
2. `git diff --stat` — every touched file is one you INTENDED to touch;
   diff size is proportionate to the task (a "small fix" touching 6 files
   is a signal-4 event).
3. Grep invariants when adapter wiring changed (00-DIAGNOSIS.md §3 L2).
4. No deleted debug prints, no reformatting, no drive-by "cleanups"
   (`git diff` shows only lines related to the task).
5. Report drafted with: what changed, verification level, what's NOT verified,
   exact user-side command if L3 is needed.

## 6. The limits — what process CANNOT compensate for (honesty clause)

Decomposition, fresh-context verification, and multi-sample review fix
EXECUTION errors. They cannot fix:
- **Ambiguous goals** ("make the adapters better") — no rubric can decide what
  the user means. Ask; don't sample-and-pick.
- **Research taste** (which adapter config is worth trying, is this metric
  believable, experiment design) — this is the user's expertise. Offer options
  with trade-offs; the user decides. A second model opinion is input, not a verdict.
- **Unknowable-from-here facts** (will training converge, GPU memory fit) —
  say plainly: "cannot be determined in this environment; here's the command
  and what to look for."
When you hit one of these: escalate to the user (preferred), or to opus for a
framed second opinion — and if it's still ambiguous, say it can't be resolved
from here. Never fabricate a confident answer to fill the gap.
