from .logos_validator_hub import validator_gate
from .async_workers import submit_async
from .config_loader import Config

class OntologyInducer:
    """
    Infers schema (types, relations, constraints) from raw inputs.
    """
    def __init__(self):
        self.config = Config()
        self.ontology = {}

    @validator_gate
    def induce(self, data: list, async_mode: bool = False):
        """
        Induce ontology from list of samples (dicts).
        async_mode: schedule in background.
        """
        if async_mode:
            submit_async(self._induce_impl, data)
            return True
        return self._induce_impl(data)

    def _induce_impl(self, data: list):
        ontology = {}
        for sample in data:
            for k, v in sample.items():
                ontology.setdefault(k, set()).add(type(v).__name__)
        self.ontology = {k: list(v) for k, v in ontology.items()}
        return self.ontology
