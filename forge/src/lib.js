// Forge shared library. build.mjs inlines this file into every workflow script, because the
// Workflow runtime has no imports. Plain JS only: no Date.now(), Math.random() or argless
// new Date() (they break resume), no filesystem access. Agents touch the disk; this file only
// decides who runs, when, with what, and what counts as done.

const FORGE_VERSION = '0.1.0'

// ---------------------------------------------------------------- config

const DEFAULTS = {
  gate: [],               // authoritative commands: run after every merge and at the end
  fastGate: [],           // cheap subset builders iterate on (falls back to gate)
  maxParallel: 8,         // concurrent builders (each gets its own worktree)
  maxTasks: 24,           // planner cap; keeps tasks reviewable
  maxRepairs: 2,          // extra build attempts per task, each with escalated effort
  maxReviewRounds: 2,     // review -> verify -> fix rounds before stopping
  reviewLenses: ['correctness', 'security', 'spec'],
  refuters: 2,            // skeptics per finding; a finding survives only on a strict majority
  fixSeverity: ['high', 'medium'],
  panel: 0,               // >= 2: that many independent plans, then a judge synthesizes
  hypotheses: 4,          // debug: parallel investigators
  fixAttempts: 2,         // debug: competing fixes, tested in parallel
  minBudget: 40000,       // with a +budget target: stop starting new work below this many tokens
  efforts: {
    scout: 'low', plan: 'high', judge: 'high', build: undefined,
    repair: ['high', 'xhigh'], integrate: 'medium', review: 'medium', refute: 'low', gate: 'low',
  },
  models: {},             // optional per-role overrides, e.g. { scout: 'haiku', judge: 'opus' }
}

function forgeConfig(a) {
  const user = (a && a.config) || {}
  return {
    ...DEFAULTS,
    ...user,
    efforts: { ...DEFAULTS.efforts, ...(user.efforts || {}) },
    models: { ...DEFAULTS.models, ...(user.models || {}) },
  }
}

// Agent options for a role. Undefined keys are dropped so the session defaults apply.
function roleOpts(c, role, extra) {
  const o = { ...(extra || {}) }
  const effort = Array.isArray(c.efforts[role]) ? c.efforts[role][0] : c.efforts[role]
  if (effort) o.effort = effort
  if (c.models[role]) o.model = c.models[role]
  return o
}

function repairEffort(c, attempt) {
  const ladder = Array.isArray(c.efforts.repair) ? c.efforts.repair : [c.efforts.repair]
  return ladder[Math.min(attempt - 1, ladder.length - 1)]
}

function canSpend(c) {
  return typeof budget === 'undefined' || !budget.total || budget.remaining() > c.minBudget
}

// ---------------------------------------------------------------- schemas

const STR = { type: 'string' }
const STRS = { type: 'array', items: STR }
const GATE = {
  type: 'object',
  properties: { pass: { type: 'boolean' }, output: STR },
  required: ['pass', 'output'],
}

const SCOUT_SCHEMA = {
  type: 'object',
  properties: {
    summary: STR,
    files: { type: 'array', items: { type: 'object', properties: { path: STR, why: STR }, required: ['path', 'why'] } },
    conventions: STRS,
    gate: STRS,
    risks: STRS,
  },
  required: ['summary', 'files', 'conventions', 'gate', 'risks'],
}

const TASK = {
  type: 'object',
  properties: {
    id: STR, title: STR, spec: STR,
    files: STRS,   // files this task owns; overlapping tasks are serialized
    deps: STRS,    // task ids that must be merged first
    accept: STRS,  // checks that prove the task is done (commands preferred)
  },
  required: ['id', 'title', 'spec', 'files', 'deps', 'accept'],
}

const PLAN_SCHEMA = {
  type: 'object',
  properties: { tasks: { type: 'array', items: TASK }, notes: STR },
  required: ['tasks', 'notes'],
}

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    summary: STR, branch: STR, commit: STR, worktree: STR,
    touched: STRS, gate: GATE,
  },
  required: ['status', 'summary', 'branch', 'commit', 'worktree', 'touched', 'gate'],
}

const MERGE_SCHEMA = {
  type: 'object',
  properties: { merged: { type: 'boolean' }, conflict: { type: 'boolean' }, commit: STR, note: STR, gate: GATE },
  required: ['merged', 'conflict', 'commit', 'note', 'gate'],
}

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: STR, line: { type: 'integer' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          title: STR, detail: STR,
        },
        required: ['file', 'line', 'severity', 'title', 'detail'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reason: STR },
  required: ['refuted', 'reason'],
}

// ---------------------------------------------------------------- plan checks (pure)

// Returns a list of problems; empty means the plan can run.
function validatePlan(tasks, maxTasks) {
  const errors = []
  if (!Array.isArray(tasks) || tasks.length === 0) return ['plan has no tasks']
  if (maxTasks && tasks.length > maxTasks) errors.push(`plan has ${tasks.length} tasks; the cap is ${maxTasks}`)
  const ids = new Set()
  for (const t of tasks) {
    if (ids.has(t.id)) errors.push(`duplicate task id ${t.id}`)
    ids.add(t.id)
  }
  for (const t of tasks) {
    for (const d of t.deps || []) if (!ids.has(d)) errors.push(`${t.id} depends on unknown task ${d}`)
    if ((t.deps || []).includes(t.id)) errors.push(`${t.id} depends on itself`)
  }
  if (!errors.length) {
    const cycle = findCycle(tasks)
    if (cycle) errors.push(`dependency cycle: ${cycle.join(' -> ')}`)
  }
  return errors
}

function findCycle(tasks) {
  const deps = new Map(tasks.map(t => [t.id, t.deps || []]))
  const state = new Map() // 1 = visiting, 2 = done
  const stack = []
  const visit = id => {
    if (state.get(id) === 2) return null
    if (state.get(id) === 1) return stack.slice(stack.indexOf(id)).concat(id)
    state.set(id, 1)
    stack.push(id)
    for (const d of deps.get(id) || []) {
      const c = visit(d)
      if (c) return c
    }
    stack.pop()
    state.set(id, 2)
    return null
  }
  for (const t of tasks) {
    const c = visit(t.id)
    if (c) return c
  }
  return null
}

function reaches(tasks, from, to) {
  const deps = new Map(tasks.map(t => [t.id, t.deps || []]))
  const seen = new Set()
  const walk = id => {
    if (id === to) return true
    if (seen.has(id)) return false
    seen.add(id)
    return (deps.get(id) || []).some(walk)
  }
  return walk(from)
}

// Two tasks that own the same file would conflict at merge time if they ran concurrently.
// Add an edge (later depends on earlier, in plan order) unless one already depends on the other.
// Adding only forward edges in plan order keeps the graph acyclic.
function serializeOverlaps(tasks) {
  const out = tasks.map(t => ({ ...t, deps: [...(t.deps || [])] }))
  const added = []
  for (let j = 1; j < out.length; j++) {
    for (let i = 0; i < j; i++) {
      const a = out[i], b = out[j]
      const shared = (a.files || []).filter(f => (b.files || []).includes(f))
      if (!shared.length) continue
      if (reaches(out, b.id, a.id) || reaches(out, a.id, b.id)) continue
      b.deps.push(a.id)
      added.push({ from: a.id, to: b.id, files: shared })
    }
  }
  return { tasks: out, added }
}

// Length of the longest dependency chain: the number of sequential build rounds the plan needs.
function criticalPath(tasks) {
  const deps = new Map(tasks.map(t => [t.id, t.deps || []]))
  const memo = new Map()
  const depth = id => {
    if (!memo.has(id)) memo.set(id, 1 + Math.max(0, ...(deps.get(id) || []).map(depth)))
    return memo.get(id)
  }
  return Math.max(0, ...tasks.map(t => depth(t.id)))
}

// ---------------------------------------------------------------- scheduling

// Limit concurrency of async jobs. Returns run(fn) -> Promise.
function semaphore(limit) {
  let active = 0
  const waiting = []
  const next = () => {
    if (active >= limit || !waiting.length) return
    active++
    const { fn, resolve, reject } = waiting.shift()
    Promise.resolve().then(fn).then(resolve, reject).finally(() => { active--; next() })
  }
  return fn => new Promise((resolve, reject) => { waiting.push({ fn, resolve, reject }); next() })
}

// Run jobs one at a time, in arrival order (the merge queue).
function mutex() {
  let tail = Promise.resolve()
  return fn => {
    const run = tail.then(() => fn())
    tail = run.catch(() => {})
    return run
  }
}

// Dependency-driven scheduler: each task starts the moment its own deps are merged, not when
// a whole wave finishes. `done` holds results for tasks finished in an earlier run (resume).
// A task whose dependency failed is skipped, not attempted.
async function runDag(tasks, runTask, limit, done) {
  const slot = semaphore(Math.max(1, limit || 1))
  const byId = new Map(tasks.map(t => [t.id, t]))
  const started = new Map()
  const start = id => {
    if (!started.has(id)) {
      const t = byId.get(id)
      const p = (done && done[id])
        ? Promise.resolve(done[id])
        : Promise.all((t.deps || []).map(start)).then(results => {
            const failed = (t.deps || []).filter((d, i) => !(results[i] && results[i].ok))
            if (failed.length) return { id, ok: false, skipped: true, reason: `dependency failed: ${failed.join(', ')}` }
            return slot(() => runTask(t)).catch(e => ({ id, ok: false, reason: String(e) }))
          })
      started.set(id, p)
    }
    return started.get(id)
  }
  return Promise.all(tasks.map(t => start(t.id)))
}

// ---------------------------------------------------------------- prompts
// Role rules live in the .claude/agents/forge-*.md system prompts (stable, so they cache).
// These builders add only the per-call facts.

function list(xs) { return (xs || []).map(x => `- ${x}`).join('\n') || '- (none)' }
function cmds(xs) { return (xs || []).map(x => `- \`${x}\``).join('\n') || '- (none configured: find and run the project\'s own tests)' }
function tail(s, n) { s = String(s || ''); return s.length > n ? '...' + s.slice(-n) : s }

function contextBlock(ctx) {
  if (!ctx) return ''
  return [
    'CONTEXT (from scouts; verify before relying on it):',
    ctx.summary || '',
    ctx.conventions && ctx.conventions.length ? 'Conventions:\n' + list(ctx.conventions) : '',
  ].filter(Boolean).join('\n')
}

function buildPrompt(t, run, prior) {
  const fast = run.c.fastGate.length ? run.c.fastGate : run.c.gate
  const deps = (t.deps || []).map(d => run.results[d]).filter(Boolean)
  return [
    `TASK ${t.id}: ${t.title}`,
    `RUN GOAL: ${run.goal}`,
    `SPEC:\n${t.spec}`,
    `OWNED FILES:\n${list(t.files)}`,
    `ACCEPTANCE:\n${list(t.accept)}`,
    `FAST GATE:\n${cmds(fast)}`,
    deps.length ? `MERGED DEPENDENCIES:\n${list(deps.map(d => `${d.id}: ${d.summary}`))}` : '',
    contextBlock(run.ctx),
    prior ? `PREVIOUS ATTEMPT FAILED. Continue from it: \`git reset --hard ${prior.commit}\` (skip if empty), then fix this:\n${tail(prior.feedback, 3000)}` : '',
  ].filter(Boolean).join('\n\n')
}

function mergePrompt(t, b, run) {
  return [
    `MERGE task ${t.id}: ${t.title}`,
    `COMMIT: ${b.commit}  BRANCH: ${b.branch}  WORKTREE: ${b.worktree}`,
    `MESSAGE: forge(${t.id}): ${t.title}`,
    `TRAILER: Forge-Task: ${t.id}`,
    `GATE:\n${cmds(run.c.gate)}`,
  ].join('\n')
}

const LENSES = {
  correctness: 'Logic errors, wrong conditions, unhandled cases, broken callers, data loss, crashes.',
  security: 'Injection, authz/authn gaps, secrets, unsafe deserialization, path traversal, SSRF.',
  spec: 'Does the change do what the goal asks? Missing pieces, half-wired features, stubs, TODOs.',
  tests: 'Behavior changes without tests that would catch a regression; tests that cannot fail.',
  perf: 'Accidental quadratic work, N+1 queries, unbounded memory, blocking calls on hot paths.',
}

function reviewPrompt(lens, run, seen) {
  return [
    `LENS: ${lens}. ${LENSES[lens] || ''}`,
    `DIFF: \`git diff ${run.base}...HEAD\``,
    `GOAL: ${run.goal}`,
    seen.length ? `ALREADY REPORTED (do not repeat):\n${list(seen.slice(-30))}` : '',
  ].filter(Boolean).join('\n\n')
}

function refutePrompt(f, run) {
  return [
    'REFUTE this finding. Default to refuted=true unless you can show a concrete input or state that triggers it in the current code.',
    `FINDING: [${f.severity}] ${f.file}:${f.line} ${f.title}\n${f.detail}`,
    `DIFF: \`git diff ${run.base}...HEAD\``,
  ].join('\n\n')
}

// ---------------------------------------------------------------- stages

// Several scouts each look a different way; none sees the others' results.
async function scout(run, extraModes) {
  const modes = [
    ['structure', 'Map the layout, entry points and conventions relevant to the goal.'],
    ['relevant', 'Find the exact files, symbols and call sites the goal will touch, and their tests.'],
  ].concat(extraModes || [])
  if (!run.c.gate.length) modes.push(['gates', 'Find the fastest commands that prove correctness here: test, typecheck, lint. Prefer targeted subsets.'])
  const out = (await parallel(modes.map(([key, ask]) => () =>
    agent(`MODE: ${key}. ${ask}\nGOAL: ${run.goal}`,
      roleOpts(run.c, 'scout', { label: `scout:${key}`, phase: 'Scout', agentType: 'forge-scout', schema: SCOUT_SCHEMA }))
  ))).filter(Boolean)
  const uniq = xs => [...new Set(xs)]
  return {
    summary: out.map(o => o.summary).join('\n'),
    files: out.flatMap(o => o.files),
    conventions: uniq(out.flatMap(o => o.conventions)).slice(0, 12),
    gate: uniq(out.flatMap(o => o.gate)),
    risks: uniq(out.flatMap(o => o.risks)).slice(0, 12),
  }
}

const PLAN_ANGLES = [
  'minimal: the fewest tasks and files that fully meet the goal',
  'risk-first: isolate risky changes, land tests and interfaces first',
  'parallel: split into many small tasks over disjoint files so builders can run at once',
]

function planPrompt(run, angle, errors) {
  return [
    `GOAL: ${run.goal}`,
    angle ? `ANGLE: ${angle}` : '',
    `LIMITS: at most ${run.c.maxTasks} tasks. Every task names the files it owns. Prefer disjoint files: tasks that share a file run one after the other.`,
    `GATE:\n${cmds(run.c.gate)}`,
    contextBlock(run.ctx),
    run.ctx && run.ctx.files.length ? `RELEVANT FILES:\n${list(run.ctx.files.slice(0, 40).map(f => `${f.path}: ${f.why}`))}` : '',
    run.ctx && run.ctx.risks.length ? `RISKS:\n${list(run.ctx.risks)}` : '',
    errors ? `YOUR LAST PLAN WAS REJECTED:\n${list(errors)}` : '',
  ].filter(Boolean).join('\n\n')
}

async function makePlan(run) {
  const c = run.c
  const plan1 = async errors => {
    if (c.panel >= 2) {
      const angles = Array.from({ length: c.panel }, (_, i) => PLAN_ANGLES[i % PLAN_ANGLES.length])
      const drafts = (await parallel(angles.map((angle, i) => () =>
        agent(planPrompt(run, angle, errors), roleOpts(c, 'plan', { label: `plan:${i + 1}`, phase: 'Plan', agentType: 'forge-scout', schema: PLAN_SCHEMA }))
      ))).filter(Boolean)
      if (drafts.length < 2) return drafts[0] || null
      return agent([
        `JUDGE ${drafts.length} plans for: ${run.goal}`,
        'Pick the one most likely to fully meet the goal with the fewest sequential steps, then graft in any task or check the others caught that it missed. Return the final plan.',
        drafts.map((d, i) => `PLAN ${i + 1} (${angles[i]}):\n${JSON.stringify(d.tasks)}`).join('\n\n'),
      ].join('\n\n'), roleOpts(c, 'judge', { label: 'plan:judge', phase: 'Plan', agentType: 'forge-scout', schema: PLAN_SCHEMA }))
    }
    return agent(planPrompt(run, null, errors), roleOpts(c, 'plan', { label: 'plan', phase: 'Plan', agentType: 'forge-scout', schema: PLAN_SCHEMA }))
  }
  let plan = await plan1(null)
  let errors = plan ? validatePlan(plan.tasks, c.maxTasks) : ['planner returned nothing']
  if (errors.length) {
    log(`plan rejected (${errors.length} problems), replanning once`)
    plan = await plan1(errors)
    errors = plan ? validatePlan(plan.tasks, c.maxTasks) : ['planner returned nothing']
  }
  if (errors.length) return { plan: null, errors }
  return { plan, errors: [] }
}

// One task: build in an isolated worktree, then merge through the queue. Every failure feeds
// the next attempt, which runs at a higher effort from where the last one stopped.
async function buildAndMerge(t, run) {
  const c = run.c
  let prior = null
  for (let attempt = 0; attempt <= c.maxRepairs; attempt++) {
    if (attempt > 0 && !canSpend(c)) { log(`${t.id}: budget low, not retrying`); break }
    const opts = roleOpts(c, 'build', {
      label: attempt ? `build:${t.id}#${attempt}` : `build:${t.id}`,
      phase: 'Build', agentType: 'forge-builder', schema: BUILD_SCHEMA, isolation: 'worktree',
    })
    if (attempt) opts.effort = repairEffort(c, attempt)
    const b = await agent(buildPrompt(t, run, prior), opts)
    if (!b) { prior = { commit: prior ? prior.commit : '', feedback: 'the previous builder died before reporting' }; continue }
    if (b.status === 'blocked') return { id: t.id, ok: false, reason: `blocked: ${b.summary}` }
    if (!b.gate.pass) { prior = { commit: b.commit, feedback: `Fast gate failed:\n${b.gate.output}` }; continue }
    const m = await run.mergeQueue(() =>
      agent(mergePrompt(t, b, run), roleOpts(c, 'integrate', { label: `merge:${t.id}`, phase: 'Integrate', agentType: 'forge-integrator', schema: MERGE_SCHEMA })))
    if (m && m.merged && m.gate.pass) {
      const r = { id: t.id, ok: true, commit: m.commit, summary: b.summary, attempts: attempt + 1 }
      run.results[t.id] = r
      log(`merged ${t.id} (${attempt + 1} attempt${attempt ? 's' : ''})`)
      return r
    }
    prior = {
      commit: b.commit,
      feedback: !m ? 'the integrator died' : m.conflict ? `Merge conflict with work merged meanwhile: ${m.note}. Rebase your change onto the current branch HEAD.` : `Gate failed after merge (merge was undone): ${m.note}\n${m.gate.output}`,
    }
  }
  log(`${t.id} failed after ${c.maxRepairs + 1} attempts`)
  return { id: t.id, ok: false, reason: tail(prior ? prior.feedback : 'no attempt ran', 600) }
}

// Group near-duplicate findings (same file, nearby line) so each bug is verified once.
function groupFindings(findings) {
  const rank = { high: 0, medium: 1, low: 2 }
  const groups = new Map()
  for (const f of findings) {
    const k = `${f.file}:${Math.floor((f.line || 0) / 5)}`
    const g = groups.get(k)
    if (!g) groups.set(k, { ...f })
    else {
      if (rank[f.severity] < rank[g.severity]) g.severity = f.severity
      if (!g.title.includes(f.title)) { g.title += ' / ' + f.title; g.detail += '\n---\n' + f.detail }
    }
  }
  return [...groups.values()]
}

function findingKey(f) { return `${f.file}:${Math.floor((f.line || 0) / 5)}` }

// Lenses find in parallel; each new finding is attacked by independent refuters as soon as its
// lens finishes (pipeline, no barrier). Survivors need a strict majority of refuters to fail.
async function reviewRound(run, seen) {
  const c = run.c
  const seenTitles = [...seen.values()]
  const perLens = await pipeline(
    c.reviewLenses,
    lens => agent(reviewPrompt(lens, run, seenTitles), roleOpts(c, 'review', { label: `review:${lens}`, phase: 'Review', agentType: 'forge-reviewer', schema: FINDINGS_SCHEMA })),
    r => groupFindings((r && r.findings) || []).filter(f => !seen.has(findingKey(f))),
    fresh => parallel(fresh.map(f => () => {
      if (!c.fixSeverity.includes(f.severity)) return Promise.resolve({ ...f, confirmed: false, skipped: 'severity' })
      return parallel(Array.from({ length: c.refuters }, (_, i) => () =>
        agent(refutePrompt(f, run), roleOpts(c, 'refute', { label: `refute:${f.file}:${f.line}#${i + 1}`, phase: 'Verify', agentType: 'forge-reviewer', schema: VERDICT_SCHEMA }))
      )).then(votes => {
        const upheld = votes.filter(v => v && !v.refuted).length
        return { ...f, confirmed: upheld * 2 > c.refuters, votes: votes.filter(Boolean).map(v => v.reason) }
      })
    })),
  )
  // Two lenses can report the same bug; keep one per key.
  const out = new Map()
  for (const f of perLens.filter(Boolean).flat().filter(Boolean)) {
    const k = findingKey(f)
    if (!out.has(k) || (f.confirmed && !out.get(k).confirmed)) out.set(k, f)
  }
  for (const [k, f] of out) seen.set(k, `${f.file}:${f.line} ${f.title}`)
  return [...out.values()]
}

// One fix task per file, so fixes never collide and all run at once.
function fixTasks(confirmed, round) {
  const byFile = new Map()
  for (const f of confirmed) byFile.set(f.file, (byFile.get(f.file) || []).concat(f))
  return [...byFile.entries()].map(([file, fs], i) => ({
    id: `fix${round}-${i + 1}`,
    title: `fix ${fs.length} review finding${fs.length > 1 ? 's' : ''} in ${file}`,
    spec: fs.map(f => `[${f.severity}] line ${f.line}: ${f.title}\n${f.detail}`).join('\n\n'),
    files: [file],
    deps: [],
    accept: ['each finding above no longer reproduces', 'gate passes'],
  }))
}

async function reviewAndFix(run, opts) {
  const c = run.c
  const seen = new Map()
  const rounds = []
  for (let round = 1; round <= c.maxReviewRounds; round++) {
    if (!canSpend(c)) { log('budget low, stopping review'); break }
    const found = await reviewRound(run, seen)
    const confirmed = found.filter(f => f.confirmed)
    rounds.push({ round, found: found.length, confirmed: confirmed.map(f => `${f.file}:${f.line} ${f.title}`) })
    log(`review round ${round}: ${found.length} new findings, ${confirmed.length} confirmed`)
    if (!confirmed.length || (opts && opts.noFix)) break
    const tasks = fixTasks(confirmed, round)
    rounds[rounds.length - 1].fixes = await runDag(tasks, t => buildAndMerge(t, run), c.maxParallel)
  }
  return rounds
}

async function finalGate(run) {
  if (!run.c.gate.length) return { pass: true, output: 'no gate configured' }
  const g = await agent(`Run each command and report. Do not modify anything.\n${cmds(run.c.gate)}`,
    roleOpts(run.c, 'gate', { label: 'gate:final', phase: 'Gate', agentType: 'forge-reviewer', schema: GATE }))
  return g || { pass: false, output: 'gate agent died' }
}

function newRun(a) {
  const c = forgeConfig(a)
  return { c, goal: a.goal || '', base: a.base || 'HEAD~1', ctx: null, results: {}, mergeQueue: mutex() }
}
