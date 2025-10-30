import pytest
from causal.scm import SCM
from agency.planner import Planner

@pytest.fixture
def planner():
    scm = SCM(dag={})
    scm.parameters = {}
    return Planner(scm=scm, rollouts=10, depth=2)

def test_plan_simple_goal(planner):
    goal = {'X': 1}
    plan = planner.plan(goal)
    assert isinstance(plan, list)
    for step in plan:
        assert isinstance(step, dict)

def test_plan_async_mode(planner):
    result = planner.plan({'Y': 0}, async_mode=True)
    assert result == []
