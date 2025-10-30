from core.logos_validator_hub import validator_gate
from core.async_workers import submit_async
from core.config_loader import Config
from datetime import datetime

class Goal:
    """
    Represents a single goal with lifecycle states and priority.
    """
    def __init__(self, name: str, priority: int = 0, horizon: int = 1):
        self.name = name
        self.priority = priority
        self.horizon = horizon  # in days
        self.created_at = datetime.utcnow()
        self.state = 'proposed'  # states: proposed, adopted, shelved, retired

class GoalManager:
    """
    Manages goal lifecycle: propose, adopt, shelve, retire, and arbitration.
    """
    def __init__(self):
        self.goals = []  # list of Goal
        self.config = Config()

    @validator_gate
    def propose_goal(self, name: str, priority: int = 0, horizon: int = 1):
        """Propose a new goal without manual schema."""
        goal = Goal(name, priority, horizon)
        self.goals.append(goal)
        return goal

    @validator_gate
    def adopt_goal(self, goal: Goal):
        """Adopt a proposed goal."""
        if goal in self.goals and goal.state == 'proposed':
            goal.state = 'adopted'
        return goal

    @validator_gate
    def shelve_goal(self, goal: Goal):
        """Temporarily shelve an adopted goal."""
        if goal in self.goals and goal.state == 'adopted':
            goal.state = 'shelved'
        return goal

    @validator_gate
    def retire_goal(self, goal: Goal):
        """Permanently retire a goal."""
        if goal in self.goals and goal.state != 'retired':
            goal.state = 'retired'
        return goal

    @validator_gate
    def list_goals(self, state: str = None):
        """List goals optionally filtered by state."""
        return [g for g in self.goals if state is None or g.state == state]
