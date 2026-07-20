# Implementation plan (template — edit for your project)

## Goal

<one paragraph: what "done" looks like for this run>

## Validation (the oracle)

Every task below is complete only when ALL of these exit 0:

```bash
npm test --silent
npm run -s typecheck
npm run -s lint
```

## Tasks

One independent, separately-verifiable task per line. Risk-tag anything that
touches auth/data/money/migrations `(risk: high)` so review routes it to a
human.

- [ ] <first task>
- [ ] <second task> (risk: high)
- [ ] <third task>
