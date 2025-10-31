
theory 3PDN_Universal_Reconciliation
  imports Main
begin

typedecl i
consts Saved :: "i ⇒ bool"
consts Reconciled :: "i ⇒ bool"

(* Premises (Section IV) *)
axiomatization where
  P1_eff_saved_all: "∀x. Saved x" and
  P3_saved_to_reconciled: "∀x. Saved x ⟶ Reconciled x"

theorem 3PDN_universal_reconciled: "∀x. Reconciled x"
  using P1_eff_saved_all P3_saved_to_reconciled by blast

end
