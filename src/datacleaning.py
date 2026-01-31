import pandas as pd
import numpy as np
import os

def run_cleaning_pipeline(input_file='.\data\original\maternal_dataset_csv.csv', output_file='.\data\processed\safemom_model_ready.csv'):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load data
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Initial raw dataset: {df.shape}")

    # Drop columns with missing values >=80%
    missing_threshold = 0.80
    limit = len(df) * (1 - missing_threshold)
    df = df.dropna(thresh=limit, axis=1)

    
    features = [
        'age', 'height_cm', 'weight_kg', 'blood_pressure_v1', 
        'hemoglobin_check_result_v1', 'no_pregnancy', 'number_of_prior_deliveries', 
        'urine_analysis_result_v1', 'syphilis_test_result_v1', 'malaria_rapid_test_result_v1', 
        'blood_grp_check_result_v1', 'check_blood_rhesus_factor_result_v1', 
        'smoking', 'use_of_alcohol', 'heart_problems', 'hypertension', 
        'vaginal_bleeding', 'total_antenatal_visits', 'mode_of_delivery', 
        'gender_of_new_born_child', 'immunization_status', 'Risk', 
        'vaginal_discharge', 'nausea_and_vomiting', 'duration_of_pregnancy_weeks_', 
        'respiratory_rate_v1', 'pulse_rate_v1', 'tetanus_immunization_status_v1', 
        'check_date_of_first_foetal_movement_v1', 'body_temperature_v1', 
        'breast_examination_v1', 'uterine_size_symphysis_fundal_height_v1', 
        'abdominal_palpation_v1'
    ]
    
    existing_cols = [c for c in features if c in df.columns]
    df = df[existing_cols].copy()

    # Numeric and Non-numeric imputation
    numeric_targets = ['age', 'height_cm', 'weight_kg', 'hemoglobin_check_result_v1', 
                       'total_antenatal_visits', 'duration_of_pregnancy_weeks_', 
                       'respiratory_rate_v1', 'pulse_rate_v1', 'body_temperature_v1']
    
    for col in df.columns:
        if col in numeric_targets:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        else:
           
            if any(key in col.lower() for key in ['result', 'test', 'blood_grp', 'check']):
                df[col] = df[col].fillna('Unknown')
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

    #FEATURE DERIVATION
    # BMI
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        df['bmi'] = df['bmi'].replace([np.inf, -np.inf], np.nan).fillna(df['bmi'].median())
    

    # Anemia Indicator
    if 'hemoglobin_check_result_v1' in df.columns:
        df['is_anemic'] = (df['hemoglobin_check_result_v1'] < 11.0)

    # Target feature cleaning
    if 'Risk' in df.columns:
        df = df.dropna(subset=['Risk'])

    # Export
    df.to_csv(output_file, index=False)
    print(f"Final features: {df.shape[1]}")
    print(f"Success! Saved to {output_file}")

if __name__ == "__main__":
    run_cleaning_pipeline()