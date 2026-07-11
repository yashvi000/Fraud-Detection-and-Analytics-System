import pytest
import pandas as pd

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "user_id" : [0, 0, 0, 0, 1, 1, 1, 1],
        "card" : [0, 0, 1, 1, 0, 0, 0, 0],
        "timestamp" : pd.to_datetime([
            "2005-01-01 10:00",
            "2005-01-01 10:30",
            "2005-01-02 01:00",
            "2011-03-25 10:00",
            "2013-01-15 09:00",
            "2013-02-01 09:05",
            "2013-03-14 09:10",
            "2013-04-05 09:30"
        ]),
        "amount" : [100.0, 200.0, 50.0, 300.0, 80.0, 90.0, 40.0, 30.0],
        "is_refund" : [0, 0, 0, 0, 0, 0, 0, 0],
        "mcc" : [5411, 5411, 5912, 5411, 5411, 5912, 7777, 5411],
        "use_chip" : ["swipe", "online", "swipe", "swipe", "swipe", "online", "swipe", "swipe"],
        "merchant_name" : ["1001", "1002", "1001", "1001", "2001", "2001", "2002", "2001"],
        "merchant_state" : ["CA", "ONLINE", "CA", "NY", "TX", "ONLINE", "TX", "TX"],
        "merchant_city" : ["LA", "ONLINE", "LA", "NYC", "Austin", "ONLINE", "Austin", "Austin"],
        "is_fraud" : [0, 0, 0, 0, 0, 0, 1, 0]
    })