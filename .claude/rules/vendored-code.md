---
paths:
  - "fairseq/**"
  - "s3prl/s3prl/upstream/**"
---
# Rules for vendored / upstream code

Most of this tree is vendored upstream code, not the project.
(Exception: `s3prl/s3prl/upstream/wav2vec2/` and `interfaces.py` are live —
`adapter-code.md` governs those.)

1. **Treat `fairseq/` as read-only.** It exists so `fairseq.checkpoint_utils`
   can load HuBERT/wav2vec2 checkpoints. If a task seems to require editing
   fairseq, stop and re-read the task — the fix almost certainly belongs in
   `s3prl/s3prl/upstream/wav2vec2/wav2vec2_model.py`, which is this fork's own
   copy of the model code. Editing fairseq is allowed only when the user names
   a fairseq file explicitly.
2. **Never touch `fairseq/.eggs/`, `fairseq/build/`, `fairseq/build.zip`.**
   They are committed build artifacts. Do not clean them up, do not add them
   to .gitignore, do not "helpfully" delete them — that is a user decision.
3. **Do not search these trees by default.** Scope Grep/Glob to the live paths
   listed in `.claude/docs/PROJECT-MAP.md`. If you must search here, do it in
   an Explore subagent, never in the main conversation.
4. Other upstream model families (`s3prl/s3prl/upstream/hubert/`, `wavlm/`,
   etc.) do NOT have adapter support. If the user asks to add adapters to a
   new upstream, the pattern to copy is in `wav2vec2/` — read
   `.claude/docs/PROJECT-MAP.md` first, and treat it as a large task
   (plan + user confirmation on scope).
