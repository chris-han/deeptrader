'''
# Data Transformation 
Verify datetime objects are correctly converted to string format 'YYYY-MM-DD'
# Happy Path 
Ensure original get_data function is called with correct parameters
# Edge Case 
Verify string dates are passed through without modification
'''

import pytest
from datetime import datetime
from case9_yahoo_fin_com_sig_tema_best import get_data_with_cache,prepare_prediction_data
from yahoo_fin.stock_info import get_data
import sys 
import os 
# Get the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
# Add the parent directory to the system path
sys.path.append(parent_dir)

@pytest.fixture
def mock_get_data(monkeypatch):
    mock = pytest.Mock()
    monkeypatch.setattr('yahoo_fin.stock_info.get_data', mock)
    return mock

def test_datetime_to_string_conversion(mock_get_data):
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    
    get_data_with_cache('AAPL', start, end)
    
    mock_get_data.assert_called_once_with(
        'AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31', 
        index_as_date=True,
        interval='1d'
    )

def test_get_data_params_passthrough(mock_get_data):
    get_data_with_cache('MSFT', '2023-01-01', '2023-12-31', 
                       index_as_date=False, interval='1wk')
    
    mock_get_data.assert_called_once_with(
        'MSFT',
        start_date='2023-01-01',
        end_date='2023-12-31',
        index_as_date=False,
        interval='1wk'
    )

def test_string_dates_unmodified(mock_get_data):
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    get_data_with_cache('GOOG', start_date, end_date)
    
    mock_get_data.assert_called_once_with(
        'GOOG',
        start_date=start_date,
        end_date=end_date,
        index_as_date=True,
        interval='1d'
    )