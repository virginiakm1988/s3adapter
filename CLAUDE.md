# s3adapter

Research fork of the s3prl speech-SSL toolkit adding parameter-efficient
fine-tuning (AdapterBias, Houlsby, LoRA, BitFit, CNN adapters) to upstream
models, plus a vendored fairseq. Single-author research code: no CI, no test
suite for the modification layer, experiments run on the user's GPU machine.

## Routing — read the matching file BEFORE acting, not after failing

Do not load these preemptively. When a trigger below matches, Read that file.

| Trigger | Read |
|---|---|
| Any task needing >2 tool calls, or your first task this session | `.claude/docs/00-DIAGNOSIS.md` |
| Locating code, "where is / how does X work" | `.claude/docs/PROJECT-MAP.md` |
| About to spawn a subagent, or task spans >3 files | `.claude/docs/10-DISPATCH.md` |
| Unsure: done? escalate? ask user? retry? | `.claude/docs/20-JUDGMENT.md` |
| Writing a subagent prompt | `.claude/docs/30-TEMPLATES.md` |
| Editing CLAUDE.md, `.claude/**`, or memory files | `.claude/docs/40-MAINTENANCE.md` |
| Session start after long gap; or things feel off | `.claude/docs/50-LETTER.md` |

Rules in `.claude/rules/` auto-load when you edit matching paths (adapter
layer, vendored code). Do not duplicate their content here.

## Facts that are always true here (memorize, don't re-derive)

1. **The project is ~5 files.** `run_downstream.py`, `downstream/runner.py`,
   `upstream/wav2vec2/wav2vec2_model.py`, `upstream/wav2vec2/expert.py`,
   `upstream/interfaces.py` (all under `s3prl/s3prl/`). The other ~3,960
   tracked files are vendored upstream code. Full map with line anchors:
   `.claude/docs/PROJECT-MAP.md`.
2. **Adapter selection is keyed on the experiment name, not the flag.**
   `'houlsby' in sys.argv[-1]` etc. — the `-n <name>` argument passed last.
   `--adapter` defaults to the STRING `"None"` and is almost decorative.
   This is intended behavior, not a bug. Details: `.claude/rules/adapter-code.md`.
3. **Nothing can run here.** No torch, no GPU, no data in this container.
   Do not `pip install torch`. Max local verification is
   `python3 -m py_compile` + grep invariants; real runs happen on the user's
   machine (verification ladder: `.claude/docs/00-DIAGNOSIS.md` §3).
4. **Disk is ephemeral; only pushed commits survive.** Commit and push to the
   session's designated branch after each completed unit of work, not once at
   the end.
5. **Search scope default:** the 5 live files + `s3prl/s3prl/downstream/`.
   Never search `fairseq/`, `fairseq/.eggs/`, `s3prl/docs/` from the main
   conversation.

## Commands

- Syntax check: `python3 -m py_compile <file>`
- Find adapter wiring: `grep -n "sys.argv\[-1\]" s3prl/s3prl/upstream/wav2vec2/wav2vec2_model.py`
- What trains: `grep -n '"adapter" in name\|requires_grad' s3prl/s3prl/downstream/runner.py`
- User-side run (L3 verify, give to user — do not run here):
  `python3 run_downstream.py --adapter houlsby -u hubert -d asr -m train -f -n <exp_name_containing_adapter_keyword>`

## Style

- Match existing code style: this codebase uses debug prints, Chinese-adjacent
  comment markers (`####`), and loose spacing. Do not reformat, do not add
  type hints, do not run formatters on files you didn't otherwise change.
- Keep diffs minimal — the user reviews raw diffs against upstream s3prl.
