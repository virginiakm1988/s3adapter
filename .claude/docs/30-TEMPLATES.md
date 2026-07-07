# Dispatch prompt templates

Copy the template, fill every `{...}` blank, delete unused optional lines.
Never dispatch without GOAL, ACCEPT, and REPORT sections (10-DISPATCH.md Rule 2).
Model/agent-type choice: 10-DISPATCH.md Rule 3.

Shared REPORT block (paste into every template):

```
REPORT (return exactly this, nothing else):
1. Conclusions as short prose with file:line references. Do not paste file
   contents; quote at most 3 lines per finding.
2. If you produced anything longer than 40 lines, write it to {output_path}
   and return the path + a 3-line summary.
3. Last line: "CONFIDENCE: high|medium|low. Not checked: {...}"
```

## 1. SEARCH (agent: Explore, model: haiku; sonnet if fuzzy)

```
GOAL: Find {what} in this repo. I need it because {why — what decision this feeds}.
CONTEXT: Live project code is only: s3prl/s3prl/run_downstream.py,
  s3prl/s3prl/downstream/, s3prl/s3prl/upstream/wav2vec2/,
  s3prl/s3prl/upstream/interfaces.py. Check .claude/docs/PROJECT-MAP.md first —
  the answer may already be there. Do NOT search fairseq/ or s3prl/docs/.
  Note: adapter behavior is keyed on substrings of sys.argv[-1], so also try
  patterns: sys.argv, houlsby, lora, bitfit, cnn_adapter, "adapter" in name.
ACCEPT: Every occurrence of {what} within the scope above is listed with
  file:line, OR you state the exact patterns+paths searched that prove absence.
REPORT: {shared block}
```

## 2. IMPLEMENT (agent: general-purpose, model: sonnet)

```
GOAL: {change}, because {user motivation}.
CONSTRAINTS: Read .claude/rules/adapter-code.md and obey its hard rules.
  Anchors: {file:line list from PROJECT-MAP.md or a prior SEARCH}.
  Touch only: {file list}. Match existing style (debug prints, #### markers,
  no reformatting). Torch is NOT installed — do not try to run the model.
ACCEPT (all must hold):
  - [ ] python3 -m py_compile passes on every changed file
  - [ ] {task-specific condition, e.g. "new branch exists at both forward
        injection points ~:3351 and ~:3387"}
  - [ ] {task-specific condition, e.g. "all new params have 'adapter' in name"}
  - [ ] git diff touches only the listed files, no unrelated lines
DO NOT: commit, push, or edit files outside the list. Leave the working tree
  for the parent to review.
REPORT: {shared block} + the git diff --stat output.
```

## 3. REFACTOR / BATCH-APPLY (agent: general-purpose, model: haiku — pattern must already be proven)

```
GOAL: Apply the following proven pattern to {file list, ≤5 files}.
PATTERN (verbatim example from the already-verified instance):
  {paste the reference diff}
ACCEPT (per file):
  - [ ] Transformation matches the pattern exactly (same shape of change)
  - [ ] python3 -m py_compile passes
  - [ ] Nothing else in the file changed
If any file doesn't fit the pattern cleanly, SKIP it and say why — do not improvise.
REPORT: {shared block} + per-file status table (applied/skipped+reason).
```

## 4. RESEARCH (agent: general-purpose, model: sonnet; use claude-code-guide for Claude-Code questions)

```
GOAL: Answer: {question}. This decides {decision it feeds}.
SOURCES: {docs/URLs/paths to consult; say "web search allowed" explicitly if it is}.
ACCEPT:
  - [ ] Every claim has a source (URL or file:line); no from-memory API claims
  - [ ] Contradictions between sources are surfaced, not silently resolved
  - [ ] Distinguishes "documented" from "inferred"
REPORT: {shared block}. Write findings >40 lines to {scratchpad}/research-{topic}.md.
```

## 5. REVIEW / ACCEPTANCE-CHECK (agent: general-purpose; haiku for read-back, opus for risky diffs)

```
GOAL: Independently verify a change you did NOT write. Do not trust the
  author's description; check the code itself.
CHECK: In {files}, verify each:
  {paste the numbered acceptance conditions from the original dispatch}
METHOD: Read the current file state. Run python3 -m py_compile. Run these
  greps: {L2 invariant greps}. For each condition answer PASS or FAIL with
  the file:line evidence; quote the violating line on FAIL.
Also flag (as WARN, not FAIL): deleted debug prints, reformatting, edits
  outside {files}, anything that violates .claude/rules/adapter-code.md.
REPORT: {shared block} + verdict line: "ACCEPT" or "REJECT: conditions {n} failed".
```

## Filling tips (for the parent)

- If you can't fill ACCEPT with concrete checks, you're not ready to dispatch —
  do a SEARCH first or ask the user (20-JUDGMENT.md §3).
- {output_path}: use the session scratchpad for throwaway artifacts; a repo
  path only for actual deliverables.
- One subtask per agent. If your GOAL contains "and also", split it.
