import pytest
from causal.scm import SCM

# Sample DAG and data
dag = {'A': [], 'B': ['A']}
data = [
    {'A': 0, 'B': 1},
    {'A': 0, 'B': 1},
    {'A': 1, 'B': 0},
]

@pytest.fixture
def scm():
    return SCM(dag=dag)

def test_fit_populates_parameters(scm):
    result = scm.fit(data)
    assert result is True
    assert 'A' in scm.parameters
    assert 'B' in scm.parameters

def test_do_creates_intervention(scm):
    new_scm = scm.fit(data) and scm.do({'A': 1})
    assert hasattr(new_scm, 'intervention')
    assert new_scm.intervention == {'A': 1}

def test_counterfactual_returns_probability(scm):
    scm.fit(data)
    prob = scm.counterfactual({'target': 'B', 'do': {'A': 0}})
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
