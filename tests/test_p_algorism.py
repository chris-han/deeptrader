import pytest
from p_algorism import count_jumps  # Replace your_module

def test_count_jumps_sample_input():
    jumps = [3, 4, 1, 2, 5, 6, 9, 0, 1, 2, 3, 1]
    start_position = 0
    expected_jumps = 4
    assert count_jumps(jumps, start_position) == expected_jumps

def test_count_jumps_stuck():
    jumps = [2, 1, 0, 2]
    start_position = 0
    expected_jumps = -1
    assert count_jumps(jumps, start_position) == expected_jumps

def test_count_jumps_exit_immediately():
    jumps = [1, 2, 3]
    start_position = 3
    expected_jumps = 0
    assert count_jumps(jumps, start_position) == expected_jumps

def test_count_jumps_single_jump():
    jumps = [3, 1, 3]
    start_position = 0
    expected_jumps = 1
    assert count_jumps(jumps, start_position) == expected_jumps
    
def test_count_jumps_edge_case():
    jumps = [2, 1, 3]
    start_position = 0
    expected_jumps = 2
    assert count_jumps(jumps, start_position) == expected_jumps

def test_count_jumps_long_path():
    jumps = [1, 2, 3, 4, 5, 6, 7, 1, 3, 10]
    start_position = 0
    expected_jumps = 5
    assert count_jumps(jumps, start_position) == expected_jumps