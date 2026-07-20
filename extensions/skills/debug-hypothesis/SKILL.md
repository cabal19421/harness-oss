---
name: debug-hypothesis
description: Hypothesis-driven debugging that resists anchoring — reproduce, read the real evidence, rank causes by prior, disprove the cheapest first, and never blame a mature dependency without proof.
---

# Hypothesis-driven debugging

A discipline for finding the *actual* cause of a failure instead of confabulating
a plausible one. It exists to counter the specific ways an LLM (or a tired human)
goes wrong while debugging: anchoring on the first idea, rationalising it instead
of testing it, looping on one dead end, and — the classic — deciding that
something core and widely-used in a mature dependency is "broken", when the bug is
almost always in the code that changed most recently.

Follow the steps in order. Do not skip to a fix.

## 1. Reproduce before theorising

- Get a **deterministic, minimal reproduction** first. If you cannot reproduce it,
  you cannot debug it — say so and gather more signal, do not guess.
- Read the **actual** error: the full stack trace, the exact failing assertion,
  the real log lines. Quote them. Do not paraphrase from memory or infer what the
  error "probably" says.

## 2. Rank hypotheses by prior — write at least three

Before proposing any fix, list **3–5 candidate causes ranked by prior probability**,
each with the single cheapest test that would disprove it. Weight the priors:

- **Highest:** code that changed most recently (your diff, the last commit, the
  new config). `git diff` / `git log -p` / `git bisect` is ground truth — reach for
  it first.
- **Medium:** first-party code you own that interacts with the changed area.
- **Lowest — treat as a last resort:** a bug in a mature, widely-used third-party
  dependency, the language runtime, or the standard library. Code that millions of
  people run daily is very unlikely to be newly broken in a common path. Blaming it
  is an extraordinary claim that needs extraordinary evidence.

Start with the **most probable** cause, not the most interesting one.

## 3. Disprove, don't confirm

- For each hypothesis, run its cheapest disproving test and record the result.
- **Instrument, don't speculate:** add a log line / assertion / breakpoint and
  *run it*. "Show me the value" beats "the value is probably X". Trust observed
  output over any mental model.
- When a hypothesis survives a real test, it earns confidence. When it fails one,
  drop it and move on — do not keep circling the same idea.

## 4. Before you blame a dependency — prove it

If you find yourself concluding a library / framework / runtime is at fault:

1. Write a **minimal standalone reproduction** that calls the dependency directly,
   outside your code. If the bug does not reproduce there, it is in *your* code.
2. Check the dependency's issue tracker / changelog for a matching known bug.
3. Only if both point at the dependency, treat it as the cause — and even then,
   prefer working around it in first-party code over patching vendored code.

A "fix" that edits `site-packages` / `node_modules` / `vendor/` or downgrades a
pinned dependency without this proof will be flagged by the harness review gate.

## 5. State the root cause with evidence

When you name the cause, cite the concrete evidence that proves it (the failing
line, the instrumented value, the bisected commit). If you cannot — you have a
guess, not a diagnosis. Say **"insufficient evidence to name a root cause; here is
the next experiment"** rather than shipping a confident guess.
