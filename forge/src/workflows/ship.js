export const meta = {
  name: 'forge-ship',
  description: 'Forge: scout, plan a task graph, build it with parallel worktree agents, merge through a gated queue, then adversarially review and fix',
  whenToUse: 'A goal that spans several files or steps. Invoked by /forge:ship and /forge:plan.',
  phases: [
    { title: 'Scout', detail: 'parallel scouts, each searching a different way' },
    { title: 'Plan', detail: 'task graph with file ownership; validated in code' },
    { title: 'Build', detail: 'one builder per task, isolated worktree, starts when its deps merge' },
    { title: 'Integrate', detail: 'serial merge queue; gate runs after every merge' },
    { title: 'Review', detail: 'one reviewer per lens over the full diff' },
    { title: 'Verify', detail: 'independent refuters per finding' },
    { title: 'Gate', detail: 'final authoritative gate' },
  ],
}

// args: { goal, base, config, planPath?, plan?, done?: string[], dryRun?: boolean }
const run = newRun(args)
const c = run.c

let plan = args.plan || null
if (!plan) {
  phase('Scout')
  run.ctx = await scout(run)
  if (!c.gate.length && run.ctx.gate.length) {
    c.gate = run.ctx.gate.slice(0, 4)
    log(`no gate configured; using what scouts found: ${c.gate.join(' && ')}`)
  }
  phase('Plan')
  const made = await makePlan(run)
  if (!made.plan) return { ok: false, stage: 'plan', errors: made.errors }
  plan = made.plan
}

const errors = validatePlan(plan.tasks, 0)
if (errors.length) return { ok: false, stage: 'plan', errors }
const { tasks, added } = serializeOverlaps(plan.tasks)
for (const e of added) log(`serialized ${e.to} after ${e.from} (both own ${e.files.join(', ')})`)
log(`${tasks.length} tasks, critical path ${criticalPath(tasks)}, up to ${c.maxParallel} builders at once`)

const planOut = { goal: run.goal, base: run.base, tasks: plan.tasks, notes: plan.notes || '' }
if (args.dryRun) return { ok: true, stage: 'plan', plan: planOut, added, criticalPath: criticalPath(tasks) }

phase('Build')
const done = {}
for (const id of args.done || []) done[id] = { id, ok: true, summary: 'merged in an earlier run', resumed: true }
Object.assign(run.results, done)
const built = await runDag(tasks, t => buildAndMerge(t, run), c.maxParallel, done)
const failed = built.filter(r => !r.ok)

let review = []
if (built.some(r => r.ok && !r.resumed)) {
  phase('Review')
  review = await reviewAndFix(run)
}

phase('Gate')
const gate = await finalGate(run)

return {
  ok: !failed.length && gate.pass,
  plan: planOut,
  tasks: built.map(r => ({ id: r.id, ok: r.ok, attempts: r.attempts, skipped: r.skipped || false, resumed: r.resumed || false, summary: r.summary, reason: r.reason })),
  review,
  gate: { pass: gate.pass, output: tail(gate.output, 2000) },
}
