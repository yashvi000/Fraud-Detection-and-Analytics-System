-- Transactions table (preprocessed transactions)
CREATE TABLE IF NOT EXISTS transactions (
	transaction_pk BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE,
    user_id INTEGER NOT NULL,
    card SMALLINT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    amount DOUBLE PRECISION,
    is_refund BOOLEAN DEFAULT FALSE,
    merchant_name BIGINT,
    merchant_city VARCHAR(100),
    merchant_state VARCHAR(50),
    zip VARCHAR(20),
    mcc SMALLINT,
    use_chip VARCHAR(20),
    errors VARCHAR(100),
    error_bad_cvv BOOLEAN DEFAULT FALSE,
    error_bad_expiration BOOLEAN DEFAULT FALSE,
    error_bad_card BOOLEAN DEFAULT FALSE,
    error_bad_pin BOOLEAN DEFAULT FALSE,
    is_high_value BOOLEAN DEFAULT FALSE,
    is_fraud BOOLEAN DEFAULT FALSE
);

-- Chronological index
CREATE INDEX IF NOT EXISTS idx_user_card_time
ON transactions(user_id, card, timestamp);

CREATE INDEX IF NOT EXISTS idx_timestamp
ON transactions(timestamp);

-- Predictions table (contains scores and outputs for each transaction)
CREATE TABLE IF NOT EXISTS predictions (
	prediction_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    predicted_at TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50),
    fraud_probability DOUBLE PRECISION,
    risk_tier VARCHAR(20),
    expected_exposure DOUBLE PRECISION,
    inference_latency DOUBLE PRECISION,
    is_alert BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction
ON predictions(transaction_id);

-- Model Versioning index
CREATE INDEX IF NOT EXISTS idx_model_ver
ON predictions(model_version);

-- Timestamp index
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
ON predictions(predicted_at DESC);

-- Fast Alert Queries index
CREATE INDEX IF NOT EXISTS idx_predictions_alert
ON predictions(is_alert, predicted_at DESC);


-- Scored txn table (stores input from demo and user txn fields)
CREATE TABLE IF NOT EXISTS scored_transactions (
    scored_id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    user_id INTEGER,
    card SMALLINT,
    timestamp TIMESTAMP NOT NULL,
    amount DOUBLE PRECISION,
    merchant_name BIGINT,
    merchant_city VARCHAR(100),
    merchant_state VARCHAR(50),
    mcc SMALLINT,
    use_chip VARCHAR(20),
    errors VARCHAR(100),
    source VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scored_txn_id
ON scored_transactions(transaction_id);

-- Audit log table (contains retraining triggers, etc)
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50),
    event_type VARCHAR(50),
    event_timestamp TIMESTAMP DEFAULT NOW(),
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_event_type
ON audit_logs(event_type, event_timestamp DESC);
