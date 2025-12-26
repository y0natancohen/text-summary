import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
with open('summaries_1k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract data points
data_points = data['data']

# Remove the 5 longest markdown_content outliers
# Sort by markdown_content length and remove top 5
data_points_sorted = sorted(data_points, key=lambda x: len(x['markdown_content']), reverse=True)
data_points_filtered = data_points_sorted[5:]  # Remove top 5 outliers

# Calculate lengths and ratios (after filtering)
markdown_lengths = [len(item['markdown_content']) for item in data_points_filtered]
summary_lengths = [len(item['summary']) for item in data_points_filtered]
ratios = [summary_len / markdown_len if markdown_len > 0 else 0
          for markdown_len, summary_len in zip(markdown_lengths, summary_lengths)]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram 1: Markdown content length
axes[0].hist(markdown_lengths, bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Markdown Content Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Markdown Content Length\n(5 longest outliers removed)')
axes[0].grid(True, alpha=0.3)

# Histogram 2: Summary length
axes[1].hist(summary_lengths, bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Summary Length (characters)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Summary Length\n(5 longest outliers removed)')
axes[1].grid(True, alpha=0.3)

# Histogram 3: Ratio (summary/markdown_content)
axes[2].hist(ratios, bins=100, edgecolor='black', alpha=0.7, color='green')
axes[2].set_xlabel('Ratio (Summary / Markdown Content)')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Summary to Markdown Ratio')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Total data points (original): {len(data_points)}")
print(f"Total data points (after removing 5 longest outliers): {len(data_points_filtered)}")
print(f"\nMarkdown Content Length Statistics:")
print(f"  Mean: {np.mean(markdown_lengths):.2f}")
print(f"  Median: {np.median(markdown_lengths):.2f}")
print(f"  Min: {np.min(markdown_lengths)}")
print(f"  Max: {np.max(markdown_lengths)}")
print(f"\nSummary Length Statistics:")
print(f"  Mean: {np.mean(summary_lengths):.2f}")
print(f"  Median: {np.median(summary_lengths):.2f}")
print(f"  Min: {np.min(summary_lengths)}")
print(f"  Max: {np.max(summary_lengths)}")
print(f"\nRatio Statistics (Summary/Markdown):")
print(f"  Mean: {np.mean(ratios):.4f}")
print(f"  Median: {np.median(ratios):.4f}")
print(f"  Min: {np.min(ratios):.4f}")
print(f"  Max: {np.max(ratios):.4f}")
#

# pip install langid pycountry pandas

import re
import pandas as pd
import langid
import pycountry

def normalize_for_langid(s: str) -> str:
    s = s or ""
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)   # remove URLs
    s = re.sub(r"@\w+", " ", s)                   # remove @handles
    s = re.sub(r"#\w+", " ", s)                   # remove hashtags
    s = re.sub(r"\s+", " ", s).strip()
    return s

def code_to_name(code: str) -> str:
    if not code:
        return "unknown"
    lang = pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
    return lang.name if lang else code

def detect_languages_langid(texts, min_chars=15):
    rows = []
    for t in texts:
        t_norm = normalize_for_langid(t)
        if len(t_norm) < min_chars:
            rows.append(("unknown", 0.0))
            continue
        code, score = langid.classify(t_norm)  # score is not a probability, but works as confidence-ish
        rows.append((code, float(score)))

    df = pd.DataFrame(rows, columns=["lang_code", "score"])
    df["lang_name"] = df["lang_code"].apply(code_to_name)

    dist = (df["lang_name"]
            .value_counts()
            .rename_axis("language")
            .reset_index(name="count"))
    return df, dist

import json

with open('summaries_1k.json', 'r') as f:
    data = json.load(f)['data']
    texts = [item['markdown_content'] for item in data]
    per_text, distribution = detect_languages_langid(texts)
    # prin all the languages and their counts
    print(distribution)



# Example:
# texts = [...]  # list of 1000 strings
# per_text, distribution = detect_languages_langid(texts)
# print(distribution.head(20))
