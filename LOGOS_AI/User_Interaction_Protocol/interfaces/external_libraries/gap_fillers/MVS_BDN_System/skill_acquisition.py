from core.logos_validator_hub import validator_gate
from core.async_workers import submit_async
from core.config_loader import Config

class SkillAcquisition:
    """
    Extracts procedures from data logs and creates callable tools.
    """
    def __init__(self):
        self.config = Config()
        self.tools = {}

    @validator_gate
    def acquire(self, logs: list, async_mode: bool = False):
        """
        Acquire new skills from logs.
        async_mode: schedule in background.
        """
        if async_mode:
            submit_async(self._acquire_impl, logs)
            return True
        return self._acquire_impl(logs)

    def _acquire_impl(self, logs: list):
        for entry in logs:
            if isinstance(entry, str) and entry.startswith('action:'):
                parts = entry.split(':', 1)[1].split(',')
                name = parts[0].strip()
                params = [p.strip() for p in parts[1:]]
                self.tools[name] = {'params': params}
        return self.tools
