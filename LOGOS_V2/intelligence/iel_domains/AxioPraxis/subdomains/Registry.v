From Coq Require Import Program.
From PXLs Require Import PXLv3.
Require Import Beauty.Spec.
(* Require Import Goodness.Spec. *)
Module Axiopraxis_OntoProps.
  (* name -> (pillar, c_value) *)
  Definition registry : list (string * string * string) := [
  ("Beauty", BeautySpec.pillar, BeautySpec.c_value)
  (* ("Goodness", GoodnessSpec.pillar, GoodnessSpec.c_value) *)
  ].
  Goal True. exact I. Qed.
End Axiopraxis_OntoProps.
