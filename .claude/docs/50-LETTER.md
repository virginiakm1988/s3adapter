# Letter to future sessions

From the Fable 5 session of 2026-07-07 that built `.claude/`. Read this when
starting after a long gap, or when the system feels like it's fighting you.

## Three things nobody asked me to tell you

### 1. Push or it never happened
This remote container is destroyed on inactivity. The filesystem, and very
likely `~/.claude` auto-memory too, do not survive. The user's mental model is
"what's on disk is safe" — here that is false. Commit + push after every
completed unit of work. If you did something valuable and the push failed,
fixing the push IS the task now (retry with backoff: 2s/4s/8s/16s). Reports of
"done" for unpushed work are false reports.

### 2. This fork is frozen around 2022 — your training memory of s3prl is wrong
Modern s3prl (the PyPI/GitHub one you remember) restructured its APIs; this
repo predates that. Do not import patterns, paths, or flags from memory of
current s3prl or fairseq docs — check THIS repo's code. Same for loralib: the
pinned usage is `lora.Linear(in, out, r=8)`, whatever newer APIs exist. The
vendored `fairseq/` is load-bearing only for `checkpoint_utils` loading; its
committed `.eggs/` and `build.zip` look like trash but are not yours to clean.
Also: git history contains a multitask feature that was added and REVERTED
(d2ebb64 → 48f8082). If asked about multitask, the code is in history, not in
the tree — don't hallucinate that it exists.

### 3. The deliverable is usually a command, not a PR
The user runs experiments on their own GPU machine. For most tasks the ideal
final report is: minimal diff, verification level reached (L1/L2 — you can
never reach L3 here), and the EXACT command to run plus the 2-3 console lines
that confirm success (e.g. the `Adapter!!` param prints from runner.py:273-285).
Optimize for "user pastes one command and knows within a minute if it worked."
Don't open PRs unasked; don't restructure research code to look like product
code — messy prints and `####` markers are the local style, and diffs against
upstream are how the user audits changes.

## The most likely way this system degrades, and the countermeasure

**Degradation path: accretion and drift, not deletion.** Concretely:
(a) code changes make PROJECT-MAP anchors stale → a session gets burned →
loses trust in the map → stops reading it → token bleed returns;
(b) sessions append hedgy, overlapping rules after every mistake instead of
fixing the one rule that failed → CLAUDE.md swells → adherence drops for ALL
rules; (c) auto-memory accumulates claims that contradict the repo files and
nobody knows which is true.

Countermeasures, in force from today:
- Anchors: fix on sight (40-MAINTENANCE.md §5). A wrong anchor is a bug with
  a 2-minute fix, not a reason to abandon the map.
- Rules: when a rule fails you, EDIT that rule (with user approval); never add
  a second rule that half-overlaps it. One concept, one home.
- Line budgets are hard: CLAUDE.md ≤100, LESSONS.md ≤150 before compression,
  docs ≤250. First reader past the limit owns proposing the cut.
- Memory: repo wins over MEMORY.md, always (40-MAINTENANCE.md §4).
- Once a quarter (or when the user says "things feel off"): re-run the
  adversarial review — one fresh subagent reads all `.claude/` files hunting
  contradictions, dead paths, stale anchors; fix findings; push.

## What I could not verify (inherited uncertainty — check before relying)

- Whether `.claude/rules/` `paths:` frontmatter gating actually fires in your
  Claude Code version — it was specified by the user, not observed by me. If
  rules seem to never load, read them explicitly via the CLAUDE.md routing
  table; that path always works.
- Whether `~/.claude/projects/-home-user-s3adapter/memory/` persists between
  sessions in this environment. I assumed NOT (see 40-MAINTENANCE.md §4).
  If you observe it persisting, note that in LESSONS.md but keep repo files
  canonical anyway.
- Runtime behavior of ANY code here — nothing was executed (no torch/GPU).
  Line anchors and control-flow claims were verified by reading, not running.

## Unfinished items

(none at hand-off — if a later session leaves work incomplete, list it here
with: state, next step, and the acceptance conditions it was working toward)
