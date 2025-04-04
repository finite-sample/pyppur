"""
Tests for the objectives module.
"""

import numpy as np
import pytest

from pyppur.objectives import Objective

def test_objective_types():
    """Test the objective types."""
    assert Objective.DISTANCE_DISTORTION == "distance_distortion"
    assert Objective.RECONSTRUCTION == "reconstruction"
    
    # Ensure they are strings
    assert isinstance(Objective.DISTANCE_DISTORTION, str)
    assert isinstance(Objective.RECONSTRUCTION, str)

def test_objective_validation():
    """Test validation of objective types."""
    # Should not raise an error for valid types
    assert Objective.DISTANCE_DISTORTION == "distance_distortion"
    assert Objective.RECONSTRUCTION == "reconstruction"
    
    # Invalid type should raise an error
    with pytest.raises(ValueError):
        Objective("invalid")