# Dispatch protocol — how the main conversation uses subagents

Applies to every session, whatever model runs the main loop.
Purpose: keep the main context small (conclusions only) and put each piece of
work on the cheapest model that can do it reliably.

## Rule 1: The commander does not take the field

The main conversation NEVER does bulk work itself. Delegate:
- **Broad search / repo scanning** ("where is X?", "find all places that Y")
  → `Explore` agent. Exception: if `PROJECT-MAP.md` already gives the anchor,
  just Read that range directly — cheaper than any agent.
- **Reading any file end-to-end that exceeds 500 lines** → subagent that
  returns a summary + `file:line` anchors.
- **Web/docs lookups** → subagent (or `claude-code-guide` agent for questions
  about Claude Code itself).
- **Batch mechanical edits across >3 files** → subagent per batch, after the
  pattern is proven on one file in the main loop.
- **Independent verification** (Rule 5) → always a fresh subagent.

The main conversation keeps for itself: talking to the user, decisions,
small targeted edits (≤3 files, anchors known), git operations, and final
synthesis.

## Rule 2: Every dispatch contains the triad

Never send a bare question. Every subagent prompt has three parts
(fill-in templates: `.claude/docs/30-TEMPLATES.md`):
1. **Goal + motivation** — what to find/do AND why the parent needs it, so the
   agent can make sane judgment calls at edges.
2. **Acceptance conditions** — a checkable list defining "done". If you cannot
   write one, you don't understand the task yet; investigate more first.
3. **Reporting format** — exactly what to return (see Rule 4).

## Rule 3: Model and effort selection

Models available via the Agent tool's `model` parameter in this environment:
`haiku`, `sonnet`, `opus`. (A `fable` option may appear in the schema; it is
not available to this account's future sessions — do not select it.)
The Agent tool has no effort parameter; effort control exists only inside
Workflow scripts, which require explicit user opt-in — default to Agent.

| Task | Agent type | Model |
|---|---|---|
| Scoped search, list occurrences, verify anchors | Explore | haiku |
| Broad/fuzzy search ("how does X flow work") | Explore | sonnet |
| Read-back / acceptance check of a finished edit | general-purpose | haiku |
| Implement a change with clear spec | general-purpose | sonnet |
| Design/plan a multi-file change | Plan | opus (sonnet if the change is routine) |
| Review a risky diff; second opinion on a judgment call | general-purpose | opus |
| Batch-apply an already-proven pattern | general-purpose | haiku |

When unsure between two tiers, take the cheaper one — the escalation path
(Rule 6) exists precisely so this is safe.

## Rule 4: Reporting contract

Subagents return CONCLUSIONS ONLY:
- findings as short prose + `file:line` references — never pasted file bodies;
- for produced artifacts (reports, long lists, generated code >40 lines):
  write to a file (scratchpad dir for throwaway, repo path for deliverables)
  and return the path plus a 3-line summary;
- a final line `CONFIDENCE: high|medium|low` plus what was NOT checked.
Put this contract in every dispatch prompt — subagents don't know it otherwise.

## Rule 5: Never self-verify

The context that produced a change will confirm its own work. Acceptance runs
in a FRESH subagent that gets the acceptance conditions but NOT the diff
narrative:
- **File edits** → read-back agent: "Read X, confirm conditions 1-3 hold,
  report any violated, quote the violating line."
- **Code changes** → run the verification ladder (`00-DIAGNOSIS.md` §3):
  py_compile + L2 invariant greps, executed by the checking agent itself.
- **High-risk judgment** (touching the sys.argv mechanism, checkpoint format,
  config values) → second opinion from an `opus` agent given the problem
  statement but not your chosen answer; compare, then decide. If they
  disagree, that's a stop-and-ask-user signal (`20-JUDGMENT.md` §3).

## Rule 6: Escalation / de-escalation ladder

- **haiku errs once** on a subtask (wrong answer, failed acceptance, empty
  result on a task that should have results) → re-dispatch to sonnet
  immediately. Do not retry haiku with a reworded prompt.
- **sonnet errs twice on the same subtask** → dispatch to opus, and include
  the full failure trace: both failed attempts, their outputs, and why each
  failed acceptance. Never make opus rediscover what already failed.
- **Hard cap: two retry rounds per subtask** across all models. After that,
  stop and report to the user what was tried and where it broke
  (`20-JUDGMENT.md` §3 has the wording).
- **De-escalation:** once a pattern is solved and verified on one instance
  (e.g. one file of a 12-file mechanical change), batch the remaining
  instances to haiku, one agent per batch of ≤5 files, each with the proven
  diff as an in-prompt example, then acceptance-check per Rule 5.

## Rule 7: Parallelize independent dispatches

Independent subagents go in ONE message (parallel tool calls). Sequential
dispatch of independent work is pure latency. Dependent work waits.

## Worked example

Task: "add a new adapter type 'prefix_conv'".
1. Main loop: Read PROJECT-MAP.md → knows the 4 wiring points already. No search needed.
2. Dispatch (sonnet, general-purpose): implement in wav2vec2_model.py per
   `.claude/rules/adapter-code.md` hard rules 3-4; acceptance = py_compile
   passes + branch added at both forward points + params contain "adapter".
3. Parallel dispatch (haiku): update README table.
4. Fresh haiku agent: acceptance read-back on both.
5. Main loop: report to user with verification level reached (L2) and the L3
   command for their GPU box.
