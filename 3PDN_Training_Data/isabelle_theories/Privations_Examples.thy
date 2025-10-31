
theory Privations_Examples
  imports Main Privations_S5_Skeleton Privations_Instances
begin

(* === Primitive signals === *)
consts BadIntent :: "i ⇒ real"
consts TruthDeviation :: "i ⇒ real"
consts CoherenceGap :: "i ⇒ real"
consts MortalityIndex :: "i ⇒ real"

axiomatization where
  BadIntent_nonneg: "∀x. 0 ≤ BadIntent x" and
  TruthDeviation_nonneg: "∀x. 0 ≤ TruthDeviation x" and
  CoherenceGap_nonneg: "∀x. 0 ≤ CoherenceGap x" and
  Mortality_nonneg: "∀x. 0 ≤ MortalityIndex x"

(* === Measures === *)
definition mu_Evil :: "i ⇒ real" where "mu_Evil x = BadIntent x"
definition mu_False :: "i ⇒ real" where "mu_False x = TruthDeviation x"
definition mu_Incoherent :: "i ⇒ real" where "mu_Incoherent x = CoherenceGap x"
definition mu_Death :: "i ⇒ real" where "mu_Death x = MortalityIndex x"

lemma mu_Evil_nonneg_inst: "∀x. 0 ≤ mu_Evil x" unfolding mu_Evil_def using BadIntent_nonneg by auto
lemma mu_False_nonneg_inst: "∀x. 0 ≤ mu_False x" unfolding mu_False_def using TruthDeviation_nonneg by auto
lemma mu_Incoherent_nonneg_inst: "∀x. 0 ≤ mu_Incoherent x" unfolding mu_Incoherent_def using CoherenceGap_nonneg by auto
lemma mu_Death_nonneg_inst: "∀x. 0 ≤ mu_Death x" unfolding mu_Death_def using Mortality_nonneg by auto

(* Dynamics *)
consts dmu_Evil_dt dmu_False_dt dmu_Incoherent_dt dmu_Death_dt :: "i ⇒ real"
axiomatization where
  descent_Evil: "∀x. Pi x ⟶ mu_Evil x > 0 ⟶ dmu_Evil_dt x < 0" and
  descent_False: "∀x. Pi x ⟶ mu_False x > 0 ⟶ dmu_False_dt x < 0" and
  descent_Incoherent: "∀x. Pi x ⟶ mu_Incoherent x > 0 ⟶ dmu_Incoherent_dt x < 0" and
  descent_Death: "∀x. Pi x ⟶ mu_Death x > 0 ⟶ dmu_Death_dt x < 0"

(* Restoration witnesses *)
axiomatization where
  restore_Evil_iff_mu_zero: "∀x. mu_Evil x = 0 ⟶ Restore_Evil x" and
  restore_False_iff_mu_zero: "∀x. mu_False x = 0 ⟶ Restore_False x" and
  restore_Incoherent_iff_mu_zero: "∀x. mu_Incoherent x = 0 ⟶ Restore_Incoherent x" and
  restore_Death_iff_mu_zero: "∀x. mu_Death x = 0 ⟶ Restore_Death x"

(* Corrective policy *)
axiomatization where corrective_policy_total:
  "∀x. (mu_Evil x > 0 ∨ mu_False x > 0 ∨ mu_Incoherent x > 0 ∨ mu_Death x > 0) ⟶ Pi x"

axiomatization where corrective_temporary: "∀x. Pi x ⟶ (∃t. True)"

end
