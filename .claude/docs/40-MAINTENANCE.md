# Maintenance protocol — how to update this system without breaking it

The `.claude/` files were built by a one-time high-capability session.
Future sessions keep them alive. This file says who may change what, how, and
how the two memory systems (repo files vs auto memory) stay consistent.

## 1. Edit permissions

| File | You may, autonomously | Only with user approval |
|---|---|---|
| `.claude/docs/LESSONS.md` | Append new lessons (format §3) | Delete others' entries |
| `.claude/docs/PROJECT-MAP.md` | Fix drifted line anchors; add anchors for new live files | Restructure; delete sections |
| `CLAUDE.md` | Fix a factually wrong path/command | Add rules; change routing; exceed 100 lines |
| `.claude/rules/*.md` | Fix a wrong line anchor | Add/remove/weaken hard rules; change `paths:` globs |
| `10/20/30-*.md` (protocols) | Fix typos, wrong tool/model names | Change any rule, threshold, or template semantics |
| `00-DIAGNOSIS.md`, `50-LETTER.md` | Nothing | Any change (historical record) |
| Auto memory `MEMORY.md` | Everything (see §4) | — |

**Backup rule:** git history is the backup. Before editing any `.claude/` or
`CLAUDE.md` file: `git status` must be clean for that file (commit pending
changes first), then edit, then commit the edit as its own commit with message
`rules: <what and why>`. Never mix rule edits into a code commit.

## 2. When you make a mistake worth remembering

Trigger: a wasted detour >10 tool calls, a wrong claim the user corrected, a
violated rule, or a surprise about how this repo works.
Action: append ONE entry to `.claude/docs/LESSONS.md` (format in that file's
header), commit, push. Do this in the same session, before ending the turn —
unwritten lessons do not exist.

Where does a lesson go LONG-TERM?
- **Fact about the repo** (anchor, behavior, gotcha) → later merged into
  `PROJECT-MAP.md` or a `.claude/rules/` file (needs user approval per §1).
- **Fact about process** (a dispatch that failed, a verification gap) →
  proposed as an edit to `10/20/30-*.md`, asked of the user.
- Until approved/merged, it lives in LESSONS.md — which is loaded on demand,
  so an unmerged lesson still helps the next session that reads it.

## 3. Compression cadence

- LESSONS.md over **150 lines** → next session that notices must propose a
  compression to the user: merge duplicate lessons into rules, delete merged
  entries, keep the file under 60 lines after compression.
- CLAUDE.md over **100 lines** or any docs file over **250 lines** → same:
  propose extraction/deletion to the user. Growth without pruning is this
  system's main failure mode (see 50-LETTER.md).
- Who: there is no daemon. The FIRST session that reads a file and notices the
  threshold crossed owns proposing the fix. Check is cheap: `wc -l` the file
  you just read.

## 4. Auto memory (`~/.claude/projects/<project>/memory/MEMORY.md`)

Claude Code auto-memory writes notes here; only MEMORY.md's first 200 lines
load at session start. Two facts govern its use HERE:
1. **This is a remote ephemeral container.** `~/.claude` state is NOT
   guaranteed to survive container reclamation. Treat auto memory as a
   session-local cache, never as the system of record.
2. **The repo files are canonical.** On any conflict between MEMORY.md and
   `CLAUDE.md`/`.claude/**`, the repo wins, and you should fix MEMORY.md on
   the spot.

Rules:
- MEMORY.md line 1 must always be:
  `Canonical rules: repo CLAUDE.md + .claude/ — on conflict, repo wins; copy keepers to .claude/docs/LESSONS.md.`
  If it's missing (fresh container), write it.
- Use MEMORY.md freely for session-scoped state: current task progress, verified
  anchors, half-finished hypotheses.
- Any insight that must outlive the session gets copied into
  `.claude/docs/LESSONS.md` and PUSHED before the turn ends. If it isn't
  pushed, assume it's gone.
- Pruning MEMORY.md is the job of whoever appends past ~150 lines: delete
  stale task-state first, never delete line 1.

## 5. Verifying an anchor fix (the most common maintenance task)

PROJECT-MAP.md anchors drift as code changes. To fix one:
1. `grep -n "<distinctive snippet from the map entry>" <file>` to find the new line.
2. Edit only the numbers in PROJECT-MAP.md; keep descriptions unless wrong.
3. Commit as `rules: refresh anchors in PROJECT-MAP (<file>)`, push.
