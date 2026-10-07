export const meta = {
  name: 'forge-review',
  description: 'Forge: review a diff through parallel lenses, attack every finding with independent refuters, optionally fix the survivors in parallel',
  whenToUse: 'Before merging a branch or PR. Invoked by /forge:review.',
  phases: [
    { title: 'Review', detail: 'one reviewer per lens' },
    { title: 'Verify', detail: 'independent refuters per finding' },
    { title: 'Build', detail: 'one fixer per file (only with fix: true)' },
    { title: 'Integrate', detail: 'serial merge queue' },
  ],
}

// args: { base, goal?, config, fix?: boolean }
const run = newRun(args)
if (!run.goal) run.goal = 'the intent stated in the commit messages of this diff'
const rounds = await reviewAndFix(run, { noFix: !args.fix })
return { base: run.base, rounds }
