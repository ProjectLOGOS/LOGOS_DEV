(*
  PraxeoPraxis Registry
*)

Require Import String.
Require Import List.

Module PraxeoPraxisRegistry.

Definition domain_name : string := "PraxeoPraxis".
Definition domain_version : string := "1.0.0".
Definition domain_description : string := 
  "IEL domain for action reasoning and practical decision-making".

Definition mapped_property : string := "Action".
Definition property_c_value : string := "0.93811+0.05540j".
Definition property_group : string := "Practical".
Definition property_order : string := "Second-Order".

Definition trinity_weights : list (string * nat) := 
  ("existence", 100) :: ("goodness", 80) :: ("truth", 70) :: nil.

Definition exported_predicates : list string := 
  "Actionable" :: "Intentional" :: "Efficacious" :: "Moral" :: "Practical" :: "Goal_Directed" :: nil.

End PraxeoPraxisRegistry.
Export PraxeoPraxisRegistry.