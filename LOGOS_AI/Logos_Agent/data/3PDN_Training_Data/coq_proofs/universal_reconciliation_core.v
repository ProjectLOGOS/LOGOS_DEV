
(* Universal Reconciliation: compact proof from bridges *)
Require Import Coq.Logic.Classical.
Require Import privations_coq.

(* General lemma: if everyone is Saved and Saved -> Reconciled, then everyone is Reconciled. *)
Theorem universal_reconciled_from_bridges :
  (forall x, Saved x) ->
  (forall x, Saved x -> Reconciled x) ->
  forall x, Reconciled x.
Proof.
  intros Hsaved Hbridge x.
  apply Hbridge, Hsaved.
Qed.

(* Corollary: uses the skeleton axioms provided earlier. *)
Corollary universal_reconciled_core : forall x, Reconciled x.
Proof.
  apply (universal_reconciled_from_bridges universal_saved saved_to_reconciled).
Qed.
