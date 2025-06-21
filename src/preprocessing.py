import pandas as pd
import numpy as np
import os

def preprocess_trends(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop('Month', axis=1)
    # Feature engineering
    df['InverseDays'] = 1 / (np.abs(df['DaysToAnniversary']) + 1)
    df['DaysSquared'] = df['DaysToAnniversary'] ** 2
    df['AnniversaryMonth'] = df['DaysToAnniversary'].between(-15, 15).astype(int)
    df['WeightedEvents'] = df['CommemorativeEvents'] / (np.abs(df['DaysToAnniversary']) + 1)
    df['EventMediaInteraction'] = df['CommemorativeEvents'] * df['NewsArticles']
    df['HasEvent'] = (df['CommemorativeEvents'] > 0).astype(int)
    df['LogEvents'] = np.log1p(df['CommemorativeEvents'])
    # Drop unused features
    df = df.drop(['DaysToAnniversary','AnniversaryMonth','EventMediaInteraction','HasEvent','LogEvents','CommemorativeEvents'], axis=1)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Construct paths relative to the script's location to avoid path errors
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.join(script_dir, '..')
    
    input_file = os.path.join(root_dir, 'data', 'jallianwala_trends.csv')
    output_file = os.path.join(root_dir, 'data', 'final_jallianwala_dataset.csv')

    preprocess_trends(input_file, output_file)
    print(f"✅ Preprocessing complete. Output saved to {os.path.relpath(output_file)}") 