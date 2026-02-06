# section_g_dataset.py
import numpy as np
import pandas as pd

# YOUR STUDENT ID
STUDENT_ID = 22424162

# Set random seed
np.random.seed(STUDENT_ID)

# Generate 24 instances
n_samples = 24

# Feature 1: Credit_Score (3 values)
credit_score = np.random.choice(['Poor', 'Fair', 'Good'], size=n_samples, p=[0.3, 0.4, 0.3])

# Feature 2: Employment (2 values)
employment = np.random.choice(['Employed', 'Unemployed'], size=n_samples, p=[0.7, 0.3])

# Feature 3: Debt_Level (3 values)
debt_level = np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.3, 0.4, 0.3])

# Class label: Default (binary)
# Rule: Poor credit + High debt + Unemployed → higher probability of default
default = []
for cs, emp, debt in zip(credit_score, employment, debt_level):
    prob_default = 0.15  # base probability
    
    if cs == 'Poor':
        prob_default += 0.25
    elif cs == 'Fair':
        prob_default += 0.10
    
    if emp == 'Unemployed':
        prob_default += 0.25
    
    if debt == 'High':
        prob_default += 0.25
    elif debt == 'Medium':
        prob_default += 0.10
    
    # Generate default with this probability
    default.append(1 if np.random.random() < prob_default else 0)

# Create DataFrame
df = pd.DataFrame({
    'Credit_Score': credit_score,
    'Employment': employment,
    'Debt_Level': debt_level,
    'Default': default
})

# Add ID column
df.insert(0, 'ID', range(1, 25))

print("="*60)
print("GENERATED DATASET (Student ID: 22424162)")
print("="*60)
print(df.to_string(index=False))

# Class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print(df['Default'].value_counts().sort_index())
print(f"\nDefault = 0: {(df['Default']==0).sum()} ({(df['Default']==0).mean()*100:.1f}%)")
print(f"Default = 1: {(df['Default']==1).sum()} ({(df['Default']==1).mean()*100:.1f}%)")

# Save to CSV
df.to_csv('my_dataset.csv', index=False)
print(f"\n✓ Dataset saved to 'my_dataset.csv'")