
theory Universal_Reconciliation_Core
  imports Main Privations_S5_Skeleton
begin

lemma universal_reconciled_from_bridges:
  assumes US: "∀x. Saved x"
  and     BR: "∀x. Saved x ⟶ Reconciled x"
  shows "∀x. Reconciled x"
  using US BR by blast

corollary universal_reconciled_core:
  "∀x. Reconciled x"
  using universal_saved saved_to_reconciled universal_reconciled_from_bridges by blast

end
