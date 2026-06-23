# run_pipeline.py
from excel_loader import load_all_gta_data
from data_validator import validate
from preprocessing import clean_historical_data
from feature_engineering import generate_features

def main():
    print("=== ÉTAPE 1 : Extraction des datasets (2022-2025) ===")
    df_raw = load_all_gta_data(data_dir="../data")
    
    print("\n=== ÉTAPE 2 : Contrôle Qualité (Quality-Gate) ===")
    report = validate(df_raw)
    summary = report.summary()
    print(f"Validation réussie ? {summary['overall_passed']}")
    if not summary['overall_passed']:
        print(f"Avertissements/Échecs constatés : {summary['failed']}")
    
    print("\n=== ÉTAPE 3 : Preprocessing ===")
    df_clean = clean_historical_data(df_raw)
    print(f"Données nettoyées. Forme du dataset : {df_clean.shape}")
    
    print("\n=== ÉTAPE 4 : Feature Engineering ===")
    df_final = generate_features(df_clean)
    print(f"Features générées. Dataset prêt pour l'entraînement : {df_final.shape}")
    
    # Sauvegarde du dataset d'entraînement final
    df_final.to_csv("../data/processed_training_data.csv", index=False)
    print("\n→ Fichier '../data/processed_training_data.csv' créé avec succès !")

if __name__ == "__main__":
    main()