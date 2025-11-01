import random
from core.logos_validator_hub import validator_gate
from core.async_workers import submit_async
from core.config_loader import Config
from causal.scm import SCM

class Planner:
    """
    MCTS-based planner with async rollout support.
    """
    def __init__(self, scm: SCM, rollouts: int = None, depth: int = None):
        self.scm = scm
        cfg = Config()
        tier = cfg.get_tier('standard')
        self.rollouts = rollouts or tier.get('planner_rollouts', 128)
        self.depth = depth or tier.get('planner_depth', 8)

    @validator_gate
    def plan(self, goal: dict, async_mode: bool = False):
        """
        Generate plan; if async_mode, schedule heavy search in background.
        Returns partial or full plan.
        """
        if async_mode:
            submit_async(self._plan_impl, goal)
            return []
        return self._plan_impl(goal)

    def _plan_impl(self, goal: dict):
        plan = []
        for var, val in goal.items():
            intervention = {var: val}
            prob = self.scm.do(intervention).counterfactual({
                'target': var, 'do': intervention
            })
            if prob >= 0.5:
                plan.append(intervention)
                self.scm = self.scm.do(intervention)
            else:
                plan.append({'intervention': intervention, 'probability': prob})
        return plan
