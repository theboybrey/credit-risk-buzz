# section_g_calculations.py
import pandas as pd
import numpy as np
from math import log2

# Load your dataset
df = pd.read_csv('my_dataset.csv')

print("="*80)
print("SECTION G: ENTROPY & INFORMATION GAIN CALCULATIONS")
print("Student ID: 22424162")
print("="*80)

# =============================================================================
# QUESTION 8: ENTROPY OF CLASS LABEL
# =============================================================================
print("\n" + "="*80)
print("QUESTION 8: ENTROPY OF CLASS LABEL")
print("="*80)

# Count class distribution
total = len(df)
count_0 = (df['Default'] == 0).sum()
count_1 = (df['Default'] == 1).sum()

p0 = count_0 / total
p1 = count_1 / total

print(f"\nClass Distribution:")
print(f"  Default = 0: {count_0}/{total} = {p0:.4f}")
print(f"  Default = 1: {count_1}/{total} = {p1:.4f}")

# Calculate entropy
if p0 > 0:
    term0 = -p0 * log2(p0)
else:
    term0 = 0

if p1 > 0:
    term1 = -p1 * log2(p1)
else:
    term1 = 0

H_default = term0 + term1

print(f"\nEntropy Calculation:")
print(f"  H(Default) = -p0*log2(p0) - p1*log2(p1)")
print(f"             = -({p0:.4f})*log2({p0:.4f}) - ({p1:.4f})*log2({p1:.4f})")

if p0 > 0:
    print(f"             = -({p0:.4f})*({log2(p0):.4f}) - ({p1:.4f})*({log2(p1):.4f})")
else:
    print(f"             = 0 - ({p1:.4f})*({log2(p1):.4f})")

print(f"             = {term0:.4f} + {term1:.4f}")
print(f"             = {H_default:.4f} bits")

print(f"\nInterpretation:")
print(f"  Maximum entropy for binary class = 1.0 bit (50:50 split)")
print(f"  Our entropy = {H_default:.4f} bits")
if H_default > 0.9:
    print(f"  → High uncertainty (classes are relatively balanced)")
elif H_default > 0.7:
    print(f"  → Moderate uncertainty (some imbalance exists)")
else:
    print(f"  → Low uncertainty (classes are imbalanced)")

# =============================================================================
# QUESTION 9: INFORMATION GAIN FOR EACH FEATURE
# =============================================================================
print("\n" + "="*80)
print("QUESTION 9: INFORMATION GAIN CALCULATIONS")
print("="*80)

def calculate_entropy(labels):
    """Calculate entropy of a label array"""
    if len(labels) == 0:
        return 0
    
    counts = labels.value_counts()
    probs = counts / len(labels)
    
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * log2(p)
    
    return entropy

def calculate_conditional_entropy(df, feature, target):
    """Calculate H(target | feature)"""
    total = len(df)
    conditional_entropy = 0
    
    print(f"\n  Conditional Entropy H({target} | {feature}):")
    print(f"  {'Value':<15} {'Count':<8} {'H(Default|value)':<20} {'Weighted':<15}")
    print(f"  {'-'*65}")
    
    for value in df[feature].unique():
        subset = df[df[feature] == value][target]
        p_value = len(subset) / total
        h_subset = calculate_entropy(subset)
        weighted = p_value * h_subset
        conditional_entropy += weighted
        
        # Show details
        counts = subset.value_counts().sort_index()
        count_str = f"{len(subset)}"
        print(f"  {value:<15} {count_str:<8} {h_subset:.4f}{'':15} {weighted:.4f}")
    
    print(f"  {'-'*65}")
    print(f"  Total Conditional Entropy: {conditional_entropy:.4f} bits")
    
    return conditional_entropy

def calculate_information_gain(df, feature, target, H_target):
    """Calculate IG for a feature"""
    print(f"\n{'='*70}")
    print(f"Feature: {feature}")
    print(f"{'='*70}")
    
    # Show value distribution
    print(f"\n  Value Distribution:")
    for value in sorted(df[feature].unique()):
        subset = df[df[feature] == value]
        count_0 = (subset[target] == 0).sum()
        count_1 = (subset[target] == 1).sum()
        total_val = len(subset)
        print(f"    {value}: Total={total_val}, Default=0: {count_0}, Default=1: {count_1}")
    
    # Calculate conditional entropy
    H_conditional = calculate_conditional_entropy(df, feature, target)
    
    # Calculate IG
    IG = H_target - H_conditional
    
    print(f"\n  Information Gain:")
    print(f"    IG({feature}) = H({target}) - H({target}|{feature})")
    print(f"    IG({feature}) = {H_target:.4f} - {H_conditional:.4f}")
    print(f"    IG({feature}) = {IG:.4f} bits")
    
    return IG, H_conditional

# Calculate IG for each feature
features = ['Credit_Score', 'Employment', 'Debt_Level']
results = {}

for feature in features:
    ig, h_cond = calculate_information_gain(df, feature, 'Default', H_default)
    results[feature] = {'IG': ig, 'H_conditional': h_cond}

# Rank features
print("\n" + "="*80)
print("FEATURE RANKING BY INFORMATION GAIN")
print("="*80)

ranked = sorted(results.items(), key=lambda x: x[1]['IG'], reverse=True)
print(f"\n{'Rank':<6} {'Feature':<20} {'Information Gain':<20}")
print(f"{'-'*50}")
for i, (feature, vals) in enumerate(ranked, 1):
    print(f"{i:<6} {feature:<20} {vals['IG']:.4f} bits")

best_feature = ranked[0][0]
print(f"\n→ ROOT NODE: {best_feature} (highest Information Gain)")

# =============================================================================
# QUESTION 10: DECISION TREE (first two levels)
# =============================================================================
print("\n" + "="*80)
print("QUESTION 10: DECISION TREE CONSTRUCTION")
print("="*80)

print(f"\nRoot Node: {best_feature}")
print(f"\nBranches:")

for value in sorted(df[best_feature].unique()):
    subset = df[df[best_feature] == value]
    count_0 = (subset['Default'] == 0).sum()
    count_1 = (subset['Default'] == 1).sum()
    total_val = len(subset)
    majority = 0 if count_0 > count_1 else 1
    
    print(f"\n  {best_feature} = {value}:")
    print(f"    Instances: {total_val}")
    print(f"    Default=0: {count_0}, Default=1: {count_1}")
    
    # Check if pure
    if count_0 == total_val:
        print(f"    → LEAF: Default = 0 (100% pure)")
    elif count_1 == total_val:
        print(f"    → LEAF: Default = 1 (100% pure)")
    else:
        # Find next best split
        print(f"    → Majority class: {majority} ({max(count_0, count_1)}/{total_val})")
        print(f"    → Could split further on next best feature...")
        
        # Calculate IG for remaining features on this subset
        remaining_features = [f for f in features if f != best_feature]
        if len(subset) > 1:
            subset_entropy = calculate_entropy(subset['Default'])
            print(f"    → Subset entropy: {subset_entropy:.4f}")

# =============================================================================
# SAVE RESULTS FOR HANDWRITTEN WORK
# =============================================================================
print("\n" + "="*80)
print("SUMMARY FOR HANDWRITTEN CALCULATIONS")
print("="*80)

print(f"\nH(Default) = {H_default:.4f} bits")
print(f"\nInformation Gains:")
for feature, vals in results.items():
    print(f"  IG({feature}) = {vals['IG']:.4f} bits")

print(f"\nRoot Node: {best_feature}")

# Save detailed results
with open('section_g_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SECTION G RESULTS - Student ID: 22424162\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"H(Default) = {H_default:.4f} bits\n")
    f.write(f"  p(Default=0) = {p0:.4f}\n")
    f.write(f"  p(Default=1) = {p1:.4f}\n\n")
    
    for feature, vals in results.items():
        f.write(f"{feature}:\n")
        f.write(f"  IG = {vals['IG']:.4f} bits\n")
        f.write(f"  H(Default|{feature}) = {vals['H_conditional']:.4f} bits\n\n")
    
    f.write(f"Root Node: {best_feature}\n")

print("\n✓ Detailed results saved to 'section_g_results.txt'")

# =============================================================================
# QUESTION 11: MODIFY ONE INSTANCE
# =============================================================================
print("\n" + "="*80)
print("QUESTION 11: DATASET MODIFICATION ANALYSIS")
print("="*80)

# Find an instance to flip (preferably one that affects the root split)
# Let's flip the first Default=1 instance to Default=0
flip_idx = df[df['Default'] == 1].index[0]

print(f"\nOriginal Instance #{flip_idx + 1}:")
print(df.loc[flip_idx])

# Create modified dataset
df_modified = df.copy()
df_modified.loc[flip_idx, 'Default'] = 1 - df_modified.loc[flip_idx, 'Default']

print(f"\nModified Instance #{flip_idx + 1}:")
print(df_modified.loc[flip_idx])

# Recalculate entropy
count_0_new = (df_modified['Default'] == 0).sum()
count_1_new = (df_modified['Default'] == 1).sum()
p0_new = count_0_new / total
p1_new = count_1_new / total

if p0_new > 0:
    term0_new = -p0_new * log2(p0_new)
else:
    term0_new = 0

if p1_new > 0:
    term1_new = -p1_new * log2(p1_new)
else:
    term1_new = 0

H_default_new = term0_new + term1_new

print(f"\nNew Entropy:")
print(f"  H'(Default) = {H_default_new:.4f} bits")
print(f"  Change: {H_default_new - H_default:+.4f} bits")

# Recalculate IG for root feature
print(f"\nRecalculating IG for {best_feature}:")
ig_new, h_cond_new = calculate_information_gain(df_modified, best_feature, 'Default', H_default_new)

print(f"\nComparison:")
print(f"  Original IG({best_feature}) = {results[best_feature]['IG']:.4f} bits")
print(f"  New IG({best_feature})      = {ig_new:.4f} bits")
print(f"  Change: {ig_new - results[best_feature]['IG']:+.4f} bits")

# Check if root changes
print(f"\nRecalculating all features with modified dataset:")
results_new = {}
for feature in features:
    ig_mod, _ = calculate_information_gain(df_modified, feature, 'Default', H_default_new)
    results_new[feature] = ig_mod

ranked_new = sorted(results_new.items(), key=lambda x: x[1], reverse=True)
new_root = ranked_new[0][0]

print(f"\n{'='*60}")
print(f"ROOT NODE ANALYSIS:")
print(f"{'='*60}")
print(f"Original root: {best_feature} (IG = {results[best_feature]['IG']:.4f})")
print(f"New root:      {new_root} (IG = {results_new[new_root]:.4f})")

if new_root == best_feature:
    print(f"\n→ Root node REMAINS THE SAME")
    print(f"  Reason: Despite the change, {best_feature} still has highest IG")
else:
    print(f"\n→ Root node CHANGED!")
    print(f"  Reason: {new_root} now has higher IG than {best_feature}")

print("\n" + "="*80)
print("CALCULATIONS COMPLETE!")
print("="*80)
print("\nFiles generated:")
print("  1. my_dataset.csv - Your unique dataset")
print("  2. section_g_results.txt - Summary of calculations")
print("\As instructed, the entropy and information gain will be handwrittedn on paper, so the printed outputs here are for verification and reference only.")