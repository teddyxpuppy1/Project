# train_diet_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save_diet_model():
    try:
        df = pd.read_csv('fitness_diet_dataset.csv')
        df.columns = df.columns.str.strip().str.lower()

        # Only rename 'dietrecommendation' if needed
        if 'dietrecommendation' in df.columns:
            df.rename(columns={'dietrecommendation': 'diet'}, inplace=True)

        # If 'goal' doesn't exist, derive it using weight and dreamweight
        if 'goal' not in df.columns and 'dreamweight' in df.columns and 'weight' in df.columns:
            df['goal'] = df.apply(lambda row: (
                'lose_weight' if row['dreamweight'] < row['weight'] else
                'build_muscle' if row['dreamweight'] > row['weight'] else
                'maintain'
            ), axis=1)

        # Drop rows with missing critical values
        df = df.dropna(subset=['gender', 'age', 'height', 'weight', 'dreamweight', 'goal', 'diet'])

        # Encode
        le_gender = LabelEncoder()
        le_goal = LabelEncoder()

        df['gender'] = le_gender.fit_transform(df['gender'])
        df['goal'] = le_goal.fit_transform(df['goal'])

        X = df[['gender', 'age', 'height', 'weight', 'dreamweight', 'goal']]
        y = df['diet']

        # Train model
        model = RandomForestClassifier()
        model.fit(X, y)

        # Save model and encoders
        joblib.dump(model, 'diet_model.pkl')
        joblib.dump(le_gender, 'le_gender.pkl')
        joblib.dump(le_goal, 'le_goal.pkl')

        print("✅ Diet model trained and saved successfully.")
        return True, "Model trained successfully!"
    except Exception as e:
        print("❌ Failed to train model:", e)
        return False, str(e)
