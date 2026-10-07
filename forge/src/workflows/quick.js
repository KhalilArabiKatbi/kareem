export const meta = {
  name: 'forge-quick',
  description: 'Forge: one builder does a small task in place, an independent agent runs the gate, repairs escalate effort',
  whenToUse: 'A small, well-understood change. Invoked by /forge:quick.',
  phases: [
    { title: 'Build', detail: 'one builder in the working tree' },
    { title: 'Gate', detail: 'independent gate run' },
    { title: 'Review', detail: 'optional single-round review' },
  ],
}

// args: { goal, base, config, review?: boolean }
const run = newRun(args)
const c = run.c
const task = { id: 'quick', title: run.goal.slice(0, 80), spec: run.goal, files: [], deps: [], accept: ['gate passes'] }

let prior = null
let result = null
for (let attempt = 0; attempt <= c.maxRepairs; attempt++) {
  if (attempt > 0 && !canSpend(c)) break
  phase('Build')
  const opts = roleOpts(c, 'build', { label: attempt ? `build#${attempt}` : 'build', phase: 'Build', agentType: 'forge-builder', schema: BUILD_SCHEMA })
  if (attempt) opts.effort = repairEffort(c, attempt)
  const b = await agent(buildPrompt(task, run, prior && { commit: '', feedback: prior }), opts)
  if (!b) { prior = 'the previous builder died before reporting'; continue }
  if (b.status === 'blocked') return { ok: false, reason: `blocked: ${b.summary}` }
  // Trust, then verify: the builder's own gate claim is not the gate.
  phase('Gate')
  const g = await finalGate(run)
  if (g.pass) { result = { ok: true, summary: b.summary, commit: b.commit, attempts: attempt + 1 }; break }
  prior = `Gate failed after your commit ${b.commit}. Fix forward on this branch:\n${g.output}`
}
if (!result) return { ok: false, reason: tail(prior, 1500) }

if (args.review) {
  c.maxReviewRounds = 1
  result.review = await reviewAndFix(run)
}
return result
