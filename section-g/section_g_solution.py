"""
Section G: Dataset Generation & Entropy Computation
Student ID: 22424162 (used as random seed)

This script generates and analyzes a unique dataset for the decision tree assignment.
"""

import numpy as np
import pandas as pd
from collections import Counter
import math

# ============================================================================
# QUESTION 7: Dataset Generation
# ============================================================================

# Set random seed using student ID
STUDENT_ID = 22424162
np.random.seed(STUDENT_ID)

# Define features with 2-3 possible values each
# Feature justification:
# - Weather: 3 values (Sunny, Cloudy, Rainy) - affects outdoor activities
# - Temperature: 2 values (Hot, Cold) - simple binary temperature classification
# - Wind: 2 values (Weak, Strong) - affects comfort level

# Generate 24 instances
n_instances = 24

weather_options = ['Sunny', 'Cloudy', 'Rainy']
temperature_options = ['Hot', 'Cold']
wind_options = ['Weak', 'Strong']
class_options = ['Yes', 'No']

# Generate random data
weather = np.random.choice(weather_options, n_instances)
temperature = np.random.choice(temperature_options, n_instances)
wind = np.random.choice(wind_options, n_instances)
class_label = np.random.choice(class_options, n_instances)

# Create DataFrame
dataset = pd.DataFrame({
    'Instance': range(1, n_instances + 1),
    'Weather': weather,
    'Temperature': temperature,
    'Wind': wind,
    'PlayTennis': class_label
})

print("=" * 70)
print("QUESTION 7: Generated Dataset")
print("=" * 70)
print(f"\nStudent ID (Seed): {STUDENT_ID}")
print(f"\nFeature Justification:")
print("- Weather (3 values: Sunny, Cloudy, Rainy): Represents atmospheric conditions")
print("- Temperature (2 values: Hot, Cold): Binary temperature classification")
print("- Wind (2 values: Weak, Strong): Wind intensity level")
print("- PlayTennis (2 values: Yes, No): Binary class label for playing tennis")
print("\n" + "=" * 70)
print("\nDataset (24 instances):")
print("=" * 70)
print(dataset.to_string(index=False))

# ============================================================================
# QUESTION 8: Entropy of Class Label
# ============================================================================

def compute_entropy(labels):
    """Compute entropy of a label distribution."""
    total = len(labels)
    counter = Counter(labels)
    entropy = 0.0

    print(f"\n  Total instances: {total}")

    for label, count in counter.items():
        if count > 0:
            prob = count / total
            log_val = math.log2(prob)
            contribution = -prob * log_val
            entropy += contribution
            print(f"  P({label}) = {count}/{total} = {prob:.6f}")
            print(f"    -P({label}) × log₂(P({label})) = -{prob:.6f} × {log_val:.6f} = {contribution:.6f}")

    return entropy

print("\n" + "=" * 70)
print("QUESTION 8: Entropy of Class Label")
print("=" * 70)

class_counts = Counter(dataset['PlayTennis'])
print(f"\nClass distribution: {dict(class_counts)}")

print("\nStep-by-step entropy calculation:")
print("-" * 40)

H_class = compute_entropy(dataset['PlayTennis'])

print(f"\n  H(PlayTennis) = {H_class:.6f} bits")

print("\n" + "-" * 40)
print("INTERPRETATION:")
print("-" * 40)
if H_class > 0.9:
    print(f"The entropy of {H_class:.4f} is close to 1.0, indicating HIGH uncertainty.")
    print("The class labels are nearly evenly distributed, making prediction difficult.")
elif H_class > 0.5:
    print(f"The entropy of {H_class:.4f} indicates MODERATE uncertainty.")
    print("There is some imbalance but still significant unpredictability.")
else:
    print(f"The entropy of {H_class:.4f} indicates LOW uncertainty.")
    print("One class dominates, making prediction relatively easier.")

# ============================================================================
# QUESTION 9: Conditional Entropy and Information Gain
# ============================================================================

def compute_conditional_entropy(dataset, feature, class_col='PlayTennis'):
    """Compute conditional entropy H(Class|Feature)."""
    total = len(dataset)
    feature_values = dataset[feature].unique()

    print(f"\n  Computing H({class_col}|{feature}):")
    print(f"  Unique values of {feature}: {list(feature_values)}")

    conditional_entropy = 0.0

    for value in sorted(feature_values):
        subset = dataset[dataset[feature] == value]
        subset_size = len(subset)
        weight = subset_size / total

        print(f"\n    For {feature} = {value}:")
        print(f"      Subset size: {subset_size}/{total} = {weight:.6f}")

        # Count class labels in subset
        class_counts = Counter(subset[class_col])
        print(f"      Class distribution: {dict(class_counts)}")

        # Compute entropy of this subset
        subset_entropy = 0.0
        for label, count in class_counts.items():
            if count > 0:
                prob = count / subset_size
                if prob > 0 and prob < 1:
                    log_val = math.log2(prob)
                    contribution = -prob * log_val
                    subset_entropy += contribution
                    print(f"        P({label}|{feature}={value}) = {count}/{subset_size} = {prob:.6f}")
                    print(f"        Contribution: -{prob:.6f} × log₂({prob:.6f}) = {contribution:.6f}")
                elif prob == 1:
                    print(f"        P({label}|{feature}={value}) = {count}/{subset_size} = 1.0 (pure subset)")

        print(f"      H({class_col}|{feature}={value}) = {subset_entropy:.6f}")

        weighted_contribution = weight * subset_entropy
        conditional_entropy += weighted_contribution
        print(f"      Weighted contribution: {weight:.6f} × {subset_entropy:.6f} = {weighted_contribution:.6f}")

    return conditional_entropy

print("\n" + "=" * 70)
print("QUESTION 9: Conditional Entropy and Information Gain")
print("=" * 70)

features = ['Weather', 'Temperature', 'Wind']
information_gains = {}

for feature in features:
    print(f"\n{'=' * 60}")
    print(f"FEATURE: {feature}")
    print("=" * 60)

    H_conditional = compute_conditional_entropy(dataset, feature)
    IG = H_class - H_conditional
    information_gains[feature] = IG

    print(f"\n  H(PlayTennis|{feature}) = {H_conditional:.6f}")
    print(f"  IG({feature}) = H(PlayTennis) - H(PlayTennis|{feature})")
    print(f"  IG({feature}) = {H_class:.6f} - {H_conditional:.6f} = {IG:.6f}")

# Rank features
print("\n" + "=" * 70)
print("FEATURE RANKING BY INFORMATION GAIN")
print("=" * 70)
ranked_features = sorted(information_gains.items(), key=lambda x: x[1], reverse=True)
print("\nRank | Feature      | Information Gain")
print("-" * 40)
for rank, (feature, ig) in enumerate(ranked_features, 1):
    print(f"  {rank}  | {feature:12} | {ig:.6f}")

# ============================================================================
# QUESTION 10: Decision Tree Construction
# ============================================================================

print("\n" + "=" * 70)
print("QUESTION 10: Decision Tree Construction")
print("=" * 70)

root_feature = ranked_features[0][0]
print(f"\nRoot Node Selection: {root_feature}")
print(f"Reason: Highest Information Gain = {information_gains[root_feature]:.6f}")

print("\n" + "-" * 40)
print("First Two Levels of Decision Tree:")
print("-" * 40)

# Split by root feature
root_values = sorted(dataset[root_feature].unique())

print(f"\nLevel 0 (Root): {root_feature}")

for value in root_values:
    subset = dataset[dataset[root_feature] == value]
    class_dist = Counter(subset['PlayTennis'])

    print(f"\n  Branch: {root_feature} = {value}")
    print(f"    Instances: {len(subset)}")
    print(f"    Class distribution: {dict(class_dist)}")

    # Check if pure
    if len(class_dist) == 1:
        majority_class = list(class_dist.keys())[0]
        print(f"    -> LEAF NODE: {majority_class} (pure)")
    else:
        # Need to split further - find best feature for this subset
        remaining_features = [f for f in features if f != root_feature]

        # Compute entropy for this subset
        subset_entropy = 0.0
        total_subset = len(subset)
        for label, count in class_dist.items():
            if count > 0:
                prob = count / total_subset
                if prob > 0 and prob < 1:
                    subset_entropy -= prob * math.log2(prob)

        # Find best feature for Level 1
        best_ig = -1
        best_feature = None

        for feature in remaining_features:
            # Compute conditional entropy for this subset
            cond_ent = 0.0
            for fval in subset[feature].unique():
                sub_subset = subset[subset[feature] == fval]
                weight = len(sub_subset) / len(subset)
                sub_class_dist = Counter(sub_subset['PlayTennis'])
                sub_entropy = 0.0
                for label, count in sub_class_dist.items():
                    if count > 0:
                        prob = count / len(sub_subset)
                        if prob > 0 and prob < 1:
                            sub_entropy -= prob * math.log2(prob)
                cond_ent += weight * sub_entropy

            ig = subset_entropy - cond_ent
            if ig > best_ig:
                best_ig = ig
                best_feature = feature

        print(f"    -> SPLIT ON: {best_feature} (IG = {best_ig:.6f})")

        # Show Level 1 branches
        for fval in sorted(subset[best_feature].unique()):
            sub_subset = subset[subset[best_feature] == fval]
            sub_class_dist = Counter(sub_subset['PlayTennis'])
            majority = max(sub_class_dist, key=sub_class_dist.get)
            print(f"        {best_feature} = {fval}: {dict(sub_class_dist)} -> {majority}")

# ASCII Tree visualization
print("\n" + "=" * 70)
print("DECISION TREE VISUALIZATION (ASCII)")
print("=" * 70)

def draw_tree(dataset, features, depth=0, prefix="", root_feature=None):
    """Draw a simple ASCII decision tree for first 2 levels."""
    if depth == 0:
        # Use the predetermined root
        feature = root_feature
        print(f"\n[{feature}]")

        for i, value in enumerate(sorted(dataset[feature].unique())):
            subset = dataset[dataset[feature] == value]
            class_dist = Counter(subset['PlayTennis'])
            is_last = (i == len(dataset[feature].unique()) - 1)
            branch = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if len(class_dist) == 1:
                majority = list(class_dist.keys())[0]
                print(f"{branch}{feature}={value} → [{majority}]")
            else:
                print(f"{branch}{feature}={value}")

                # Find best feature for level 1
                remaining = [f for f in features if f != feature]
                subset_ent = 0
                for label, count in class_dist.items():
                    p = count/len(subset)
                    if 0 < p < 1:
                        subset_ent -= p * math.log2(p)

                best_f, best_ig = None, -1
                for f in remaining:
                    cond = 0
                    for v in subset[f].unique():
                        ss = subset[subset[f] == v]
                        w = len(ss)/len(subset)
                        e = 0
                        for l, c in Counter(ss['PlayTennis']).items():
                            p = c/len(ss)
                            if 0 < p < 1:
                                e -= p * math.log2(p)
                        cond += w * e
                    ig = subset_ent - cond
                    if ig > best_ig:
                        best_ig, best_f = ig, f

                if best_f:
                    print(f"{child_prefix}[{best_f}]")
                    for j, v2 in enumerate(sorted(subset[best_f].unique())):
                        ss2 = subset[subset[best_f] == v2]
                        cd2 = Counter(ss2['PlayTennis'])
                        maj2 = max(cd2, key=cd2.get)
                        is_last2 = (j == len(subset[best_f].unique()) - 1)
                        b2 = "└── " if is_last2 else "├── "
                        print(f"{child_prefix}{b2}{best_f}={v2} → [{maj2}] {dict(cd2)}")

draw_tree(dataset, features, root_feature=root_feature)

# ============================================================================
# QUESTION 11: Modify One Instance and Recompute
# ============================================================================

print("\n" + "=" * 70)
print("QUESTION 11: Modify One Instance and Recompute")
print("=" * 70)

# Create modified dataset
modified_dataset = dataset.copy()

# Find an instance to modify that will have impact
# Change instance 1's class label
original_instance = modified_dataset.loc[0].copy()
print(f"\nOriginal Instance 1:")
print(f"  {dict(original_instance)}")

# Flip the class label
old_class = modified_dataset.loc[0, 'PlayTennis']
new_class = 'No' if old_class == 'Yes' else 'Yes'
modified_dataset.loc[0, 'PlayTennis'] = new_class

print(f"\nModified Instance 1:")
print(f"  PlayTennis changed from '{old_class}' to '{new_class}'")

# Recompute entropy
print("\n" + "-" * 40)
print("Recomputing Class Entropy:")
print("-" * 40)

new_class_counts = Counter(modified_dataset['PlayTennis'])
print(f"New class distribution: {dict(new_class_counts)}")

H_class_new = compute_entropy(modified_dataset['PlayTennis'])
print(f"\nNew H(PlayTennis) = {H_class_new:.6f}")
print(f"Original H(PlayTennis) = {H_class:.6f}")
print(f"Change in entropy: {H_class_new - H_class:.6f}")

# Recompute Information Gains
print("\n" + "-" * 40)
print("Recomputing Information Gains:")
print("-" * 40)

new_information_gains = {}

for feature in features:
    # Quick computation
    H_cond = 0
    for value in modified_dataset[feature].unique():
        subset = modified_dataset[modified_dataset[feature] == value]
        weight = len(subset) / len(modified_dataset)
        class_dist = Counter(subset['PlayTennis'])
        subset_ent = 0
        for label, count in class_dist.items():
            p = count / len(subset)
            if 0 < p < 1:
                subset_ent -= p * math.log2(p)
        H_cond += weight * subset_ent

    IG = H_class_new - H_cond
    new_information_gains[feature] = IG

    print(f"\nIG({feature}):")
    print(f"  Original: {information_gains[feature]:.6f}")
    print(f"  New:      {IG:.6f}")
    print(f"  Change:   {IG - information_gains[feature]:.6f}")

# New ranking
new_ranked = sorted(new_information_gains.items(), key=lambda x: x[1], reverse=True)
old_root = ranked_features[0][0]
new_root = new_ranked[0][0]

print("\n" + "-" * 40)
print("NEW FEATURE RANKING:")
print("-" * 40)
print("\nRank | Feature      | Original IG  | New IG")
print("-" * 50)
for rank, (feature, ig) in enumerate(new_ranked, 1):
    print(f"  {rank}  | {feature:12} | {information_gains[feature]:.6f}    | {ig:.6f}")

print("\n" + "-" * 40)
print("ROOT NODE ANALYSIS:")
print("-" * 40)
print(f"Original root node: {old_root}")
print(f"New root node: {new_root}")

if old_root == new_root:
    print(f"\nThe root node REMAINS THE SAME ({new_root}).")
    print("Explanation: Although the information gains changed slightly,")
    print(f"the relative ranking of features remained unchanged.")
    print(f"{new_root} still has the highest information gain.")
else:
    print(f"\nThe root node CHANGED from {old_root} to {new_root}.")
    print("Explanation: The modification shifted the class distribution")
    print("enough to change which feature provides the most information.")
    print(f"Before: IG({old_root}) = {information_gains[old_root]:.6f}")
    print(f"After:  IG({new_root}) = {new_information_gains[new_root]:.6f}")

# ============================================================================
# QUESTION 12: Written Analysis (300-400 words)
# ============================================================================

print("\n" + "=" * 70)
print("QUESTION 12: Analysis of Information Gain Bias")
print("=" * 70)

analysis = """
WHY INFORMATION GAIN CAN BE BIASED:

Information Gain exhibits a well-documented bias toward features with many distinct
values. This occurs because features with more categories can create more subsets,
each potentially more "pure" than fewer, larger subsets. In the extreme case, a
feature with unique values for each instance would achieve perfect information gain
by creating completely pure singleton subsets—yet this feature would have no
predictive power on new data.

In our dataset, Weather has 3 values while Temperature and Wind each have 2. If
Weather shows higher Information Gain, part of this advantage may stem from its
additional category rather than genuine predictive utility. The bias becomes more
pronounced with larger value sets, making it problematic for features like IDs,
timestamps, or high-cardinality categorical variables.

ALTERNATIVE CRITERIA BEHAVIOR:

GAIN RATIO: This metric normalizes Information Gain by the intrinsic information
(entropy) of the feature itself. For our dataset, Gain Ratio would compute
IG/SplitInfo, where SplitInfo is higher for features with more values. Weather's
Gain Ratio would be divided by a larger denominator than Temperature or Wind,
potentially reducing its advantage. This could result in a different root node
selection if Weather's raw IG advantage stemmed primarily from having more categories.

GINI INDEX: Unlike entropy-based measures, Gini Index measures impurity as the
probability of misclassification. It tends to isolate the most frequent class in
its own branch and often produces more balanced splits. For our binary
classification problem, Gini might favor features that cleanly separate the
majority class, potentially selecting a different splitting attribute than
Information Gain would choose.

PREFERRED CRITERION FOR THIS DATASET:

Given our dataset's characteristics—three categorical features with 2-3 values
each—Gain Ratio would be the preferred criterion. The potential bias of Information
Gain toward Weather (3 values) is mitigated by Gain Ratio's normalization. While
Gini Index would also work well and is computationally simpler, Gain Ratio's
explicit correction for multi-valued features makes it more theoretically
appropriate when feature cardinalities differ.

Additionally, Gain Ratio's behavior is well-understood and widely implemented in
algorithms like C4.5, making results more reproducible and comparable to
established literature. For larger datasets or features with highly unequal
cardinalities, this correction becomes essential for building generalizable
decision trees.
"""

print(analysis)

# Word count
words = len(analysis.split())
print(f"\n[Word count: {words}]")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("COMPLETE DATASET FOR REFERENCE")
print("=" * 70)
print("\nOriginal Dataset:")
print(dataset.to_string(index=False))

# Save to CSV
dataset.to_csv('/Users/theboybrey/dev/workspace/cscd/observatory/section_g_dataset.csv', index=False)
print("\nDataset saved to: section_g_dataset.csv")
