(*
  ZelosPraxis Core - Zeal and Passionate Devotion Domain
*)

Require Import String.

Module ZelosPraxis.

Parameter Zealous : Type -> Prop.
Parameter Fervent : Type -> Prop.
Parameter Passionate : Type -> Prop.
Parameter Devoted : Type -> Prop.
Parameter Intense : Type -> Prop.

Parameter NecessarilyZealous : Type -> Prop.
Parameter PossiblyPassionate : Type -> Prop.

Axiom zealous_implies_passionate : forall x, Zealous x -> Passionate x.
Axiom fervent_implies_intense : forall x, Fervent x -> Intense x.

Definition ZealousPerfect (x : Type) : Prop :=
  Zealous x /\ Fervent x /\ Passionate x /\ Devoted x /\ Intense x.

Parameter zeal_c_value : Complex.t.
Axiom zeal_c_value_def : zeal_c_value = (0.47018 + 0.79962 * Complex.i).

Parameter zeal_trinity_weight : R * R * R.
Axiom zeal_trinity_weight_def : zeal_trinity_weight = (0.8, 0.9, 0.7).

Parameter zeal_intensity : Type -> R.
Axiom zeal_intensity_bounds : forall x, (0 <= zeal_intensity x <= 1)%R.

End ZelosPraxis.
Export ZelosPraxis.