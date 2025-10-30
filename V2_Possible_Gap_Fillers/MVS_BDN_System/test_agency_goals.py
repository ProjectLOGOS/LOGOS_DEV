import pytest
from agency.goals import GoalManager, Goal

@pytest.fixture
def manager():
    return GoalManager()

def test_propose_and_list_goals(manager):
    g = manager.propose_goal('test_goal', priority=5, horizon=2)
    assert isinstance(g, Goal)
    goals = manager.list_goals()
    assert g in goals
    assert g.state == 'proposed'

def test_adopt_shelve_retire(manager):
    g = manager.propose_goal('g1')
    manager.adopt_goal(g)
    assert g.state == 'adopted'
    manager.shelve_goal(g)
    assert g.state == 'shelved'
    manager.retire_goal(g)
    assert g.state == 'retired'
