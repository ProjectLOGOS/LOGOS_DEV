
# 3PDN Universal Reconciliation — Privation Calculus Build Guide

## Files
- `privations_coq.v` — Base skeleton: predicates, Saved→Reconciled, universal_saved axiom, universal_reconciled theorem.
- `privations_filled_coq.v` — Your μ₁…μ₈ definitions, bounds, cross-couplings, ordinal Π schedule.
- `privations_termination_coq.v` — Well-founded termination scaffold (ordinal, non-preemptive Π).
- `privations_full_coq.v` — Remaining μ₉…μ₁₅ definitions with descent and restoration hooks.
- `Privations_S5_Skeleton.thy` — Isabelle skeleton aligned with S5 bridge.
- `Privations_Instances.thy` — Instance scaffolds.
- `Privations_Examples.thy` — Example bindings for four privations.
- `Privations_Filled.thy` — Filled μ₁…μ₈ and constraints.
- `Privations_Termination.thy` — Termination scaffold in Isabelle.
- `Privations_Full.thy` — Full μ₁…μ₁₅ in Isabelle with descent/restoration hooks.

## Coq Build Order
1. `privations_coq.v`
2. `privations_filled_coq.v`
3. `privations_termination_coq.v`
4. `privations_full_coq.v`

Goal available: `universal_reconciled` (from skeleton). Replace axioms with lemmas as you define concrete dynamics.

## Isabelle Build Order
1. `Privations_S5_Skeleton.thy`
2. `Privations_Instances.thy`
3. `Privations_Examples.thy` (optional)
4. `Privations_Filled.thy`
5. `Privations_Termination.thy`
6. `Privations_Full.thy`

## Notes
- Π scheduling: ordinal, discrete, hierarchical, non-preemptive — already encoded.
- Cross-couplings: meta-bound and cascades included. Extend the 15×15 coupling matrix if needed.
- To go fully proof-complete, replace axioms labeled “Axiom/axiomatization” and “Admitted/sorry” with your concrete proofs.
