import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_utils import (
    compute_temp_features,
    compute_time_since_last_txn,
    compute_mcc_encoding,
    apply_mcc_encoding,
    compute_velocity_features,
    compute_spend_features,
    compute_zscore,
    compute_is_new_merchant,
    compute_cross_card_features,
    compute_is_new_state,
    compute_is_new_city,
    compute_cold_start_values,
    apply_cold_start_values
)

# Loading config and paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
V1_TRAIN_END = config["splits"]["v1_train_end"]


class TestTemporalFeatures:

    def test_columns_produced(self, sample_df):
        df = compute_temp_features(sample_df.copy())
        expected = [
            "day_of_week", 
            "is_weekend", 
            "is_night",
            "hour_sin",
            "hour_cos"
            ]
        
        for col in expected:
            assert col in df.columns, f"Missing column : {col}"
    
    def test_value_ranges(self, sample_df):
        df = compute_temp_features(sample_df.copy())
        assert df["day_of_week"].between(0, 6).all()
        assert df["is_weekend"].isin([0, 1]).all()
        assert df["is_night"].isin([0, 1]).all()
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_output_dtypes(self, sample_df):
        df = compute_temp_features(sample_df.copy())
        assert df["day_of_week"].dtype == "int8"
        assert df["is_weekend"].dtype == "int8"
        assert df["is_night"].dtype == "int8"
        assert df["hour_sin"].dtype == "float32"
        assert df["hour_cos"].dtype == "float32"
    
    def test_cyclic_encoding_continuity(self):
        # Hour 23 and Hour 0 are closer to each other than Hour 23 and Hour 12

        hour_23_sin = np.sin(2 * np.pi * 23 / 24)
        hour_0_sin = np.sin(2 * np.pi * 0 / 24)
        hour_12_sin = np.sin(2 * np.pi * 12 / 24)
        assert abs(hour_23_sin - hour_0_sin) < abs(hour_23_sin - hour_12_sin)

    def test_is_night_flags_late_hours(self, sample_df):
        # is_night should be 1 for transactions at hour <= 5 and >= 22

        night_df = sample_df.copy()
        night_df["timestamp"] = pd.to_datetime(["2005-01-01 23:00"] * len(night_df))
        df = compute_temp_features(night_df)
        assert (df["is_night"] == 1).all()

    def test_is_night_flags_day_hours(self, sample_df):
        # is_night should be 0 for transactions at mid-day

        day_df = sample_df.copy()
        day_df["timestamp"] = pd.to_datetime(["2005-01-01 12:00"] * len(day_df))
        df = compute_temp_features(day_df)
        assert (df["is_night"] == 0).all()

    
class TestTimeSinceLastTxn:

    def test_first_txn_per_card(self, sample_df):
        # 1st transaction per user per card should be -1

        df = sample_df
        df["minutes_since_last_txn"] = compute_time_since_last_txn(df)
        first_rows = df.groupby(["user_id", "card"]).head(1)
        assert (first_rows["minutes_since_last_txn"] == -1).all()

    def test_non_first_txn(self, sample_df):
        # All non-first transaction values should be > 0 (minutes)

        df = sample_df
        df["minutes_since_last_txn"] = compute_time_since_last_txn(df)
        non_first = (
            df.groupby(["user_id", "card"])
            .apply(lambda x: x.iloc[1:], include_groups=False)
            .reset_index(drop=True)
        )
        assert (non_first["minutes_since_last_txn"] > 0).all()

    def test_output_never_less_than_minus_one(self, sample_df):
        # Checking sort order (minutes_since_last_txn should never be < -1)
        
        result = compute_time_since_last_txn(sample_df)
        assert (result >= -1).all(), "Timestamps not sorted in chronological order"

    def test_output_dtypes(self, sample_df):
        df = sample_df
        result = compute_time_since_last_txn(df)
        assert result.dtype == "float32"
    
    def test_output_value(self, sample_df):
        # For user 0, card 0 : Row 0- 10:00, Row 1- 10:30
        # Row 1 minutes_since_last_txn should be 30

        df = sample_df[
            (sample_df["user_id"] == 0) & (sample_df["card"] == 0)
        ].copy()

        df["minutes_since_last_txn"] = compute_time_since_last_txn(df)
        assert abs(df.loc[1, "minutes_since_last_txn"] - 30.0) < 0.01


class TestMCCEncoding:

    def test_encoding_uses_only_training_data(self, sample_df, tmp_path, monkeypatch):
        # MCC encodings should be computed on Training Data only
        
        import src.features.feature_utils as fu
        monkeypatch.setattr(fu, "MCC_ENCODING_PATH", tmp_path / "mcc.pkl")

        train_df = sample_df[sample_df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        mcc_freq = compute_mcc_encoding(train_df)

        # Post-Training MCC
        assert 7777 not in mcc_freq, "Post-Training MCC should not appear"

        # Training MCC
        for mcc in train_df["mcc"].unique():
            assert mcc in mcc_freq, f"Training MCC {mcc} missing"

    def test_unknown_mcc_is_zero(self, sample_df, tmp_path, monkeypatch):
        # MCC 7777 (Post-Training) should be 0

        import src.features.feature_utils as fu
        monkeypatch.setattr(fu, "MCC_ENCODING_PATH", tmp_path / "mcc.pkl")

        train_df = sample_df[sample_df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        mcc_freq = compute_mcc_encoding(train_df)

        # Applying Training MCC Encoding to full data
        result = apply_mcc_encoding(sample_df, mcc_freq)

        post_train_mask = sample_df["timestamp"].dt.year > V1_TRAIN_END
        assert (result[
            post_train_mask & (sample_df["mcc"] == 7777)
            ] == 0).all(), "Unseen MCC frequency in training (7777) should be 0"
        
    def test_known_mcc_is_non_zero(self, sample_df, tmp_path, monkeypatch):
        # Seen MCC in Training should be positive

        import src.features.feature_utils as fu
        monkeypatch.setattr(fu, "MCC_ENCODING_PATH", tmp_path / "mcc.pkl")

        train_df = sample_df[sample_df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        mcc_freq = compute_mcc_encoding(train_df)

        # Checking on Training data only
        result = apply_mcc_encoding(train_df, mcc_freq)
        assert (result > 0).any(), "Known MCC in Training should be non-zero"

    def test_output_dtype(self, sample_df, tmp_path, monkeypatch):
        
        import src.features.feature_utils as fu
        monkeypatch.setattr(fu, "MCC_ENCODING_PATH", tmp_path / "mcc.pkl")

        train_df = sample_df[sample_df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        mcc_freq = compute_mcc_encoding(train_df)

        result = apply_mcc_encoding(sample_df, mcc_freq)
        assert result.dtype == "float32"
    

class TestLeakagePrevention:

    def test_velocity_first_row_is_zero(self, sample_df):
        # First transaction should be 0 (no history yet)

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        group = df.groupby(["user_id", "card"], sort=False)

        df["card_txn_count_60min"] = (
            group[["timestamp", "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=60), include_groups=False)
            .reset_index(level= [0, 1], drop=True)
            .astype("float32")
        )

        first_rows = df.groupby(["user_id", "card"]).head(1)
        assert (first_rows["card_txn_count_60min"] == 0).all()
    
    def test_velocity_counts_previous_txn_only(self, sample_df):
        # Current transaction should not be included

        df = (
            sample_df[sample_df["user_id"] == 0]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        group = df.groupby(["user_id"], sort=False)

        result = (
            group[["timestamp", "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=30 * 1440), include_groups=False)
            .reset_index(level=0, drop=True)
        )
        result = result.T.squeeze()
        df["user_txn_count_30d"] = result.astype("float32")
        
        assert df.loc[0, "user_txn_count_30d"] == 0
        assert df.loc[1, "user_txn_count_30d"] == 1

    def test_spend_mean_previous_txn_only(self, sample_df):
        # Current transaction should not be included in calculation
        # For user 1, card 0 : 
        # Row 7 spend mean should include only rows 4, 5 and 6 [(80 + 90 +40)/3 = 70]
        # If Row 7 amount (30) is included, mean = (80 + 90 + 40 + 30)/4 = 60

        df = (
            sample_df[
                (sample_df["user_id"] == 1) & (sample_df["card"] == 0)
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        group = df.groupby(["user_id", "card"], sort=False)

        spend = (
            group
            .apply(lambda x: compute_spend_features(x, "card", 365), include_groups=False)
            .reset_index(level= [0, 1], drop=True)
            )
        
        df[spend.columns] = spend

        row_4_amount = df.loc[0, "amount"]   # 80.00
        row_5_amount = df.loc[1, "amount"]   # 90.00
        row_6_amount = df.loc[2, "amount"]   # 40.00
        sum_amount = row_4_amount + row_5_amount + row_6_amount
        
        row_7_amount = df.loc[3, "amount"]   # 30.00
        row_7_mean = df.loc[3, "card_spend_mean_365d"]

        assert abs((sum_amount / 3) - row_7_mean) < 0.01, (
            f"Row 7 mean ({row_7_mean:.2f}) should be equal to sum of Row 4, 5, and 6 amounts ({sum_amount:.2f})"
            f"If current transaction was included, mean would be {(sum_amount + row_7_amount) / 4:.2f}"
        )
    
    def test_spend_first_row_is_null_before_cold_start(self, sample_df):
        # 1st transaction per user per card should be NaN before cold-start

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        group = df.groupby(["user_id", "card"], sort=False)

        spend = (
            group
            .apply(lambda x: compute_spend_features(x, "card", 365), include_groups=False)
            .reset_index(level= [0, 1], drop=True)
            )
        
        df[spend.columns] = spend

        first_row = df.groupby(["user_id", "card"]).head(1)
        assert first_row["card_spend_mean_365d"].isna().all()
    

class TestZScore:
        # z-score = (amount - mean) / std

    def test_equal_amount_and_mean_has_zero_zscore(self, sample_df):
        # if amount = mean, z-score should be 0

        df = sample_df.copy()
        df["user_spend_mean_30d"] = df["amount"]
        df["user_spend_std_30d"] = 13.0
        
        result = compute_zscore(df, "user", 30)
        assert (result == 0).all()

    def test_division_by_zero_std_is_not_nan_or_inf(self, sample_df):
        # 0 std should not give NaN, or inf

        df = sample_df.copy()
        df["user_spend_mean_365d"] = 25.00
        df["user_spend_std_365d"] = 0.00

        result = compute_zscore(df, "user", 365)
        assert not result.isna().any()
        assert not result.isin([np.inf, -np.inf]).any()

    def test_clipped_at_boundary(self, sample_df):
        # Should be clipped at [-10, 10]

        df = sample_df.copy()
        df["card_spend_mean_30d"] = 0.0
        df["card_spend_std_30d"] = 1.0
        df["amount"] = 100.00

        result = compute_zscore(df, "card", 30)
        assert (result <= 10).all()
        assert (result >= -10).all()

    def test_output_dtype(self, sample_df):

        df = sample_df.copy()
        df["card_spend_mean_365d"] = 50.00
        df["card_spend_std_365d"] = 10.00

        result = compute_zscore(df, "card", 365)
        assert result.dtype == "float32"
    

class TestMerchantFamiliarity:

    def test_first_txn_per_card_is_new(self, sample_df):
        # 1st transaction per user per card is always a new merchant
        # 1st transaction should have value 1 (new)

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        merch_df = df[["user_id", "card", "merchant_name"]].copy()

        result = compute_is_new_merchant(merch_df, "card", ["user_id", "card"])
        first_txn_idx = df.groupby(["user_id", "card"]).head(1).index
        assert (result.loc[first_txn_idx] == 1).all()
    
    def test_repeated_merchant_is_not_new(self, sample_df):
        # Repeated merchants should have value 0 (not new)
        # for user 0, card 1 : merchant 1001 is repeated in row 3 (1st txn in row 2)

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        merch_df = df[["user_id", "card", "merchant_name"]].copy()
        result_card = compute_is_new_merchant(merch_df, "card", ["user_id", "card"])

        assert result_card.iloc[0] == 1, "First occurrence of 1001 (Row 0) should be 1 (user 0, card 0)"
        assert result_card.iloc[2] == 1, "First occurence of 1001 (Row 2) should be 1 (user 0, card 1)" 
        assert result_card.iloc[3] == 0, "Second occurence of 1001 (Row 3) should be 0 (user 0, card 1)" 
    
    def test_user_level_includes_all_cards(self, sample_df):
        # User-level groups by user and not card
        # for user, merchant 1001 is repeated at Row 2 (1st txn in row 0)

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        merch_df = df[["user_id", "card", "merchant_name"]].copy()
        result_user = compute_is_new_merchant(merch_df, "user", ["user_id"])

        assert result_user.iloc[0] == 1, "First occurrence of 1001 (Row 0) should be 1 (user 0)"
        assert result_user.iloc[2] == 0, "Second occurence of 1001 (Row 2) should be 0 (user 0)"

    def test_output_dtype(self, sample_df):
        merch_df = sample_df[["user_id", "card", "merchant_name"]].copy()
        result = compute_is_new_merchant(merch_df, "card", ["user_id", "card"])
        assert result.dtype == "int8"


class TestCrossCardFeatures:

    def test_first_txn_per_user_is_zero(self, sample_df):

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        group = df[["user_id", "card", "timestamp"]].groupby("user_id", sort=False)
        
        result = compute_cross_card_features(group, window_min=1440)
        first_txn_idx = df.groupby("user_id").head(1).index
        assert (result.loc[first_txn_idx] == 0).all()
    
    def test_distinct_cards_counted_correctly(self):
        # user 0 uses 2 distinct cards (0, 1) within the window of 1440 minutes
        # Row 0 -> first txn -> 0
        # Row 1 -> card 0 used before -> 1 card
        # Row 2 -> same as row 1
        # Row 3 -> cards 0 and 1 used before -> 2 distinct cards
        # Row 4 -> no txn before in 1440 min window -> 0

        df = pd.DataFrame({
            "user_id"  : [0, 0, 0, 0, 0],
            "card"     : [0, 0, 1, 0, 2],
            "timestamp": pd.to_datetime([
                "2005-01-01 10:00",
                "2005-01-01 10:30",
                "2005-01-01 11:00",
                "2005-01-02 03:30",
                "2005-03-15 09:45"
            ]),
        })
        group = df.groupby("user_id", sort=False)
        
        result = compute_cross_card_features(group, window_min=1440)
        assert result.iloc[0] == 0, "First transactions per user should be 0 (Row 0)"
        assert result.iloc[1] == 1, "One card (0) used before (Row 1)"
        assert result.iloc[2] == 1, "One card (0) used before (Row 2)"
        assert result.iloc[3] == 2, "Two cards (0, 1) used before (Row 3)"
        assert result.iloc[4] == 0, "No transaction before in 1440 minute window (Row 4)"
    
    def test_output_dtype(self, sample_df):
        df = sample_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        group = df[["user_id", "card", "timestamp"]].groupby("user_id", sort=False)

        result = compute_cross_card_features(group, window_min=1440)
        assert result.dtype == "int8"


class TestGeographicFeatures:

    def test_online_is_zero(self, sample_df):
        # ONLINE txn should be 0 for merchant_state and merchant_city

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        geo_df = df[["user_id", "card", "merchant_state", "merchant_city"]].copy()

        result_state = compute_is_new_state(geo_df, "card", ["user_id", "card"])
        result_city = compute_is_new_city(geo_df, "card", ["user_id", "card"])
        
        online_state_mask = df["merchant_state"] == "ONLINE"
        online_city_mask = df["merchant_city"] == "ONLINE"
        
        assert (result_state[online_state_mask] == 0).all()
        assert (result_city[online_city_mask] == 0).all()

    def test_is_new_state_output(self, sample_df):
        # 1st non-ONLINE state should be 1 (new)
        # Repeated states should be 0 (not new)
        
        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        geo_df = df[["user_id", "card", "merchant_state"]].copy()
        result = compute_is_new_state(geo_df, "card", ["user_id", "card"])
        
        assert result.iloc[0] == 1, "First occurence of 'CA' in Row 0 (user 0, card 0)"
        assert result.iloc[2] == 1, "First occurence of 'CA' in Row 2 (user 0, card 1)"
        assert result.iloc[4] == 1, "First occurence of 'TX' in Row 4 (user 1, card 0)"
        assert result.iloc[6] == 0, "Second occurence of 'TX' in Row 6 (user 1, card 0)"

    def test_is_new_city_output(self, sample_df):
        # 1st non-ONLINE city should be 1 (new)
        # Repeated cities should be 0 (not new)
        
        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        geo_df = df[["user_id", "card", "merchant_city"]].copy()
        result = compute_is_new_city(geo_df, "card", ["user_id", "card"])
        
        assert result.iloc[0] == 1, "First occurence of 'LA' in Row 0 (user 0, card 0)"
        assert result.iloc[2] == 1, "First occurence of 'LA' in Row 2 (user 0, card 1)"
        assert result.iloc[4] == 1, "First occurence of 'Austin' in Row 4 (user 1, card 0)"
        assert result.iloc[6] == 0, "Second occurence of 'Austin' in Row 6 (user 1, card 0)"
    
    def test_output_dtype(self, sample_df):
        
        geo_df = sample_df[["user_id", "card", "merchant_state", "merchant_city"]].copy()
        result_state = compute_is_new_state(geo_df, "card", ["user_id", "card"])
        result_city = compute_is_new_city(geo_df, "card", ["user_id", "card"])

        assert result_state.dtype == "int8"
        assert result_city.dtype == "int8"
    

class TestColdStartFillValues:

    def test_no_nulls_after_fill(self, sample_df):
        # There should be no NaN after Cold-Start Handling

        df = sample_df.copy()
        df["card_spend_mean_30d"] = np.nan
        df["card_txn_count_60min"] = np.nan

        fill_values = compute_cold_start_values(df, [30], [60], save_path=None)
        df = apply_cold_start_values(df, fill_values)

        assert df["card_spend_mean_30d"].isna().sum() == 0
        assert df["card_txn_count_60min"].isna().sum() == 0
    
    def test_zero_fills(self, sample_df):
        # Velocity and Z-Score fill values should be 0
        
        fill_values = compute_cold_start_values(sample_df, [30], [60], save_path=None)
        
        assert fill_values["card_txn_count_60min"] == 0
        assert fill_values["card_txn_count_30d"] == 0
        assert fill_values["card_amount_zscore_30d"] == 0
        
    def test_spend_fills_from_training(self, sample_df):
        # Spend mean fills should be equal to training mean
        # Spend std fills should be equal to training std

        train_df = sample_df[sample_df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        train_mean = train_df["amount"].mean()
        train_std = train_df["amount"].std()

        train_df["user_spend_mean_30d"] = np.nan
        train_df["user_spend_std_30d"] = np.nan
        fill_values = compute_cold_start_values(train_df, [30], [60], save_path=None)
        
        assert abs(fill_values["user_spend_mean_30d"] - train_mean) < 0.01
        assert abs(fill_values["user_spend_std_30d"] - train_std) < 0.01
    

class TestIntegration:

    def test_full_pipeline(self, sample_df, tmp_path, monkeypatch):
        # There should be no nulls
        # Row counts should be same
        # All features should be present

        import src.features.feature_utils as fu
        monkeypatch.setattr(fu, "MCC_ENCODING_PATH", tmp_path / "mcc.pkl")
        monkeypatch.setattr(fu, "COLD_START_PATH", tmp_path / "cold_start.pkl")

        df = sample_df.sort_values(["user_id", "card", "timestamp"]).reset_index(drop=True)
        n_rows = len(df)

        # Temporal
        df = compute_temp_features(df)
        df["minutes_since_last_txn"] = compute_time_since_last_txn(df)

        # MCC Encoding (training data)
        train_df = sample_df[sample_df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        mcc_freq = compute_mcc_encoding(train_df)
        df["mcc_frequency"] = apply_mcc_encoding(df, mcc_freq)


        # Card Features
        group_cards = df.groupby(["user_id", "card"], sort=False)
        
        df["card_txn_count_60min"] = (
            group_cards[["timestamp", "is_refund"]]
            .apply(lambda x: compute_velocity_features(x, window_min=60), include_groups=False)
            .reset_index(level= [0, 1], drop=True)
            .astype("float32")
        )

        spend = (
            group_cards
            .apply(lambda x: compute_spend_features(x, "card", 365), include_groups=False)
            .reset_index(level= [0, 1], drop=True)
            )
        df[spend.columns] = spend


        # Z-Score Features
        df[f"card_amount_zscore_365d"] = compute_zscore(df, "card", 365)
        df = df.drop(columns=["card_spend_mean_365d", "card_spend_std_365d"])

        # Merchant Familiarity Features
        merch_df = df[["user_id", "card", "merchant_name"]].copy()
        df["card_is_new_merchant"] = compute_is_new_merchant(merch_df, 'card', ['user_id', 'card'])
        
        # Cross-Card Level Features
        cross_card_df = df[["user_id", "card", "timestamp"]].copy()
        group = cross_card_df.groupby("user_id", sort=False)

        df[f"distinct_cards_used_1440min"] = (
            compute_cross_card_features(group, window_min=1440)
            .fillna(0)
            .astype("int8")
        )
        
        # Geographical Features
        geo_card_df = df[["user_id", "card", "merchant_state", "merchant_city"]].copy()
        df["card_is_new_state"] = compute_is_new_state(geo_card_df, 'card', ['user_id', 'card'])
        df["card_is_new_city"] = compute_is_new_city(geo_card_df, 'card', ['user_id', 'card'])
        
        # Online Transactions Flag
        df["is_online"] = (df["use_chip"] == "online").astype("int8")

        # Cold-Start Handling
        train_df = df[df["timestamp"].dt.year <= V1_TRAIN_END].copy()
        fill_values = compute_cold_start_values(train_df, [365], [60], save_path=None)
        df = apply_cold_start_values(df, fill_values)
        
        # Assertions
        assert df.isna().sum().sum() == 0   # No remaining nulls
        assert len(df) == n_rows   # Row count remains same

        expected_cols = [
            "day_of_week", "is_weekend", "is_night", "hour_sin", "hour_cos",
            "minutes_since_last_txn", "mcc_frequency", "card_txn_count_60min", 
            "card_amount_zscore_365d", "card_is_new_merchant", "is_online", 
            "distinct_cards_used_1440min", "card_is_new_state", "card_is_new_city"
        ]

        for col in expected_cols:
            assert col in df.columns, f"Expected column {col} missing from pipeline output"