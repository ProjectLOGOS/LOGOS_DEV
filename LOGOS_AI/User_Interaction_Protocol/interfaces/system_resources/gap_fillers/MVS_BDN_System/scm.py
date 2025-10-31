from core.logos_validator_hub import validator_gate
from core.async_workers import submit_async
from core.config_loader import Config

class SCM:
    """
    Structural Causal Model with async fit capability.
    """
    def __init__(self, dag=None):
        self.dag = dag or {}
        self.parameters = {}
        self.config = Config()

    @validator_gate
    def fit(self, data: list, async_mode: bool = False):
        """
        Fit structural equations; run async if heavy.
        data: list of observations.
        async_mode: if True, schedule in background and return immediately.
        """
        tier = self.config.get_tier('standard')
        max_iter = tier.get('max_fractal_iter', 500)
        if async_mode:
            submit_async(self._fit_impl, data, max_iter)
            return True
        else:
            return self._fit_impl(data, max_iter)

    def _fit_impl(self, data: list, max_iter: int):
        from collections import defaultdict
        counts = {}
        for node, parents in self.dag.items():
            counts[node] = defaultdict(lambda: defaultdict(int))
            for sample in data[:max_iter]:
                key = tuple(sample[p] for p in parents) if parents else ()
                val = sample.get(node)
                counts[node][key][val] += 1
            self.parameters[node] = {
                key: {v: c/sum(freq.values()) for v, c in freq.items()}
                for key, freq in counts[node].items()
            }
        return True

    @validator_gate
    def do(self, intervention: dict):
        new = SCM(dag=self.dag)
        new.parameters = dict(self.parameters)
        new.intervention = intervention
        return new

    @validator_gate
    def counterfactual(self, query: dict):
        target = query.get('target')
        do = query.get('do', {})
        if target in do:
            return 1.0
        params = self.parameters.get(target, {})
        if not params:
            return 0.0
        return sum(sum(dist.values()) for dist in params.values()) / (len(params) * len(next(iter(params.values()))))
