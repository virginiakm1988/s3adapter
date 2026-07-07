# Lessons log (append-only; see 40-MAINTENANCE.md §2-3 for rules)

Entry format — copy exactly, one entry per lesson, newest at top:

```
## YYYY-MM-DD <one-line title>
- Situation: <1-2 lines, what task>
- Mistake/surprise: <what went wrong or what was unexpected>
- Rule to apply next time: <imperative, checkable>
- Merge target: PROJECT-MAP | rules/adapter-code | rules/vendored-code | 10-DISPATCH | 20-JUDGMENT | none
```

---

## 2026-07-07 LoRA gating is on sys.argv[-1] AND applies to k/q (not q/v)
- Situation: building PROJECT-MAP.md, first draft claimed LoRA replaced q/v
  projections unconditionally.
- Mistake/surprise: wrote a plausible-from-memory fact without reading the
  guard 10 lines above the grep hit. Actual: `if 'lora' in sys.argv[-1]:`
  wraps k_proj and q_proj replacement (wav2vec2_model.py:846-853).
- Rule to apply next time: before recording any claim about adapter wiring,
  read 10 lines ABOVE the grep hit — every injection here is wrapped in a
  sys.argv guard.
- Merge target: none (already merged into PROJECT-MAP.md).
