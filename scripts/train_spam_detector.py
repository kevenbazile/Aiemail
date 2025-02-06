import os
import pandas as pd

# ✅ Get the Correct Path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'email.csv')  # ✅ Use Absolute Path

# ✅ Check if File Exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found: {data_path}. Ensure 'email.csv' is inside 'data/'.")

# ✅ Load the Dataset
df = pd.read_csv(data_path, encoding='latin-1')[['text', 'label']]
print("✅ `email.csv` loaded successfully!")
