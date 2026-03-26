-- ============================================================
-- Hearing Loss Prediction System — MySQL Schema
-- ============================================================
-- Run this file once to initialise the database:
--   mysql -u root -p < database/schema.sql

CREATE DATABASE IF NOT EXISTS hearing_loss_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE hearing_loss_db;

-- ----------------------------------------------------------
-- 1. Patient demographic information
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS patients (
    id             INT            AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(100)   NOT NULL,
    age            INT            NOT NULL,
    gender         TINYINT(1)     NOT NULL COMMENT '0=Female, 1=Male',
    noise_exposure FLOAT          DEFAULT 0  COMMENT 'Years of noise exposure',
    tinnitus       TINYINT(1)     DEFAULT 0  COMMENT '0=No, 1=Yes',
    created_at     DATETIME       DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------
-- 2. Raw audiometric measurements per patient
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS audiometric_data (
    id                   INT   AUTO_INCREMENT PRIMARY KEY,
    patient_id           INT   NOT NULL,

    -- Left ear thresholds (dB HL)
    hearing_250hz_left   FLOAT, hearing_500hz_left   FLOAT,
    hearing_1000hz_left  FLOAT, hearing_2000hz_left  FLOAT,
    hearing_4000hz_left  FLOAT, hearing_8000hz_left  FLOAT,

    -- Right ear thresholds (dB HL)
    hearing_250hz_right  FLOAT, hearing_500hz_right  FLOAT,
    hearing_1000hz_right FLOAT, hearing_2000hz_right FLOAT,
    hearing_4000hz_right FLOAT, hearing_8000hz_right FLOAT,

    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- ----------------------------------------------------------
-- 3. Model prediction results
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id             INT            AUTO_INCREMENT PRIMARY KEY,
    patient_id     INT            NOT NULL,
    severity_label VARCHAR(20)    NOT NULL COMMENT 'Normal/Mild/Moderate/Severe',
    severity_class TINYINT        NOT NULL COMMENT '0-3',
    confidence     FLOAT          NOT NULL COMMENT 'Max class probability',
    input_source   VARCHAR(10)    DEFAULT 'manual' COMMENT 'manual or audio',
    predicted_at   DATETIME       DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);
