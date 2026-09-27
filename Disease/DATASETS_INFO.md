# Integrated Datasets Information

## Overview
The Parkinson's Disease Detection System integrates multiple research datasets for comprehensive analysis. All datasets are available in CSV format in the `datasets/` directory.

## Available Datasets

### 1. Voice Dataset (`voice_dataset.csv`)

**Source:** UCI Oxford Parkinson's Disease Dataset

**Description:**
- 195 voice recordings from 31 individuals
- 23 biomarkers per recording
- Binary classification (PD vs. Healthy)

**Features:**
- **Fundamental Frequency Measures:**
  - `MDVP:Fo(Hz)` - Average vocal fundamental frequency
  - `MDVP:Fhi(Hz)` - Maximum vocal fundamental frequency
  - `MDVP:Flo(Hz)` - Minimum vocal fundamental frequency

- **Jitter Measures (frequency variation):**
  - `MDVP:Jitter(%)` - Relative jitter
  - `MDVP:Jitter(Abs)` - Absolute jitter
  - `MDVP:RAP` - Relative amplitude perturbation
  - `MDVP:PPQ` - Five-point period perturbation quotient
  - `Jitter:DDP` - Average absolute difference

- **Shimmer Measures (amplitude variation):**
  - `MDVP:Shimmer` - Relative shimmer
  - `MDVP:Shimmer(dB)` - Shimmer in decibels
  - `Shimmer:APQ3` - Three-point amplitude perturbation
  - `Shimmer:APQ5` - Five-point amplitude perturbation
  - `MDVP:APQ` - 11-point amplitude perturbation
  - `Shimmer:DDA` - Average absolute difference

- **Harmonic Measures:**
  - `NHR` - Noise-to-harmonics ratio
  - `HNR` - Harmonics-to-noise ratio

- **Nonlinear Measures:**
  - `RPDE` - Recurrence period density entropy
  - `DFA` - Detrended fluctuation analysis
  - `spread1`, `spread2` - Nonlinear fundamental frequency variation
  - `D2` - Correlation dimension
  - `PPE` - Pitch period entropy

**Target Variable:**
- `status` - 1 = Parkinson's Disease, 0 = Healthy

**Sample Size:**
- Total: 195 recordings
- PD patients: 147 recordings
- Healthy controls: 48 recordings

**Usage:**
```python
import pandas as pd
df = pd.read_csv('datasets/voice_dataset.csv')
print(df.head())
```

---

### 2. Handwriting Dataset (`handwriting_dataset.csv`)

**Description:**
- 77 handwriting samples
- 15 motor function features
- Spiral and wave pattern analysis

**Features:**
- `tremor_frequency` - Frequency of tremor in Hz (4-8 Hz range)
- `tremor_amplitude` - Amplitude of tremor oscillations (0-1 scale)
- `line_smoothness` - Smoothness of drawn lines (0-1, higher is smoother)
- `velocity_variation` - Variation in drawing speed (0-1)
- `stroke_width` - Width of pen strokes in pixels
- `size_consistency` - Consistency of drawing size (0-1)
- `writing_speed` - Speed of writing in pixels/second
- `pressure_variation` - Variation in pen pressure (0-1)
- `spatial_coverage` - How much of canvas is used (0-1)
- `drawing_density` - Density of ink on canvas (0-1)
- `symmetry_score` - Symmetry of drawings (0-1)
- `completion_rate` - How complete the drawing is (0-1)
- `spiral_regularity` - Regularity of spiral patterns (0-1)
- `micrographia_score` - Degree of micrographia (0-1, higher indicates more micrographia)
- `rhythm_score` - Rhythmic consistency (0-1)

**Target Variable:**
- `class` - 1 = Parkinson's Disease, 0 = Healthy

**Sample Size:**
- Total: 77 subjects
- PD patients: 62 subjects
- Healthy controls: 15 subjects

**Clinical Significance:**
- Micrographia (small handwriting) is a common PD symptom
- Tremor patterns visible in spiral drawings
- Reduced smoothness and increased variation indicate motor dysfunction

**Usage:**
```python
import pandas as pd
df = pd.read_csv('datasets/handwriting_dataset.csv')
print(df.describe())
```

---

### 3. PPMI Clinical Dataset (`ppmi_clinical_dataset.csv`)

**Source:** Parkinson's Progression Markers Initiative (PPMI)

**Description:**
- 600 clinical instances
- 15 clinical progression markers
- Comprehensive disease assessment

**Features:**
- `age_at_enrollment` - Patient age at study enrollment (years)
- `disease_duration` - Years since PD diagnosis
- `updrs_total` - Unified Parkinson's Disease Rating Scale total score
- `updrs_motor` - UPDRS motor examination score
- `updrs_bradykinesia` - Bradykinesia (slowness) subscore
- `updrs_tremor` - Tremor subscore
- `updrs_rigidity` - Rigidity subscore
- `moca_score` - Montreal Cognitive Assessment score (0-30)
- `semantic_fluency` - Verbal fluency test score
- `datscan_binding` - DaTscan imaging binding ratio
- `csf_alpha_synuclein` - CSF alpha-synuclein level (pg/mL)
- `csf_tau` - CSF tau protein level (pg/mL)
- `led_total` - Levodopa equivalent daily dose (mg)
- `schwab_england` - Activities of Daily Living scale (0-100%)
- `hoehn_yahr_stage` - Disease severity stage (0-5)

**Target Variable:**
- `group` - "PD" or "Control"

**Sample Size:**
- Total: 600 instances
- PD patients: 400 instances
- Healthy controls: 200 instances

**Clinical Scales:**

**UPDRS (Unified Parkinson's Disease Rating Scale):**
- 0-199 points total
- Higher scores indicate worse symptoms
- Motor section: 0-108 points

**MoCA (Montreal Cognitive Assessment):**
- 0-30 points
- ≥26 considered normal
- <26 suggests cognitive impairment

**Hoehn & Yahr Staging:**
- Stage 0: No signs of disease
- Stage 1: Unilateral symptoms
- Stage 2: Bilateral symptoms
- Stage 3: Postural instability
- Stage 4: Severe disability
- Stage 5: Wheelchair/bed-bound

**Schwab & England ADL:**
- 100%: Completely independent
- 70-80%: Not completely independent
- 50-60%: More dependent
- <50%: Severely dependent

**Usage:**
```python
import pandas as pd
df = pd.read_csv('datasets/ppmi_clinical_dataset.csv')
pd_patients = df[df['group'] == 'PD']
print(pd_patients[['updrs_total', 'hoehn_yahr_stage']].describe())
```

---

## Data Analysis Examples

### Example 1: Voice Feature Comparison
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/voice_dataset.csv')

# Compare jitter between PD and healthy
pd_jitter = df[df['status'] == 1]['MDVP:Jitter(%)']
healthy_jitter = df[df['status'] == 0]['MDVP:Jitter(%)']

plt.boxplot([healthy_jitter, pd_jitter], labels=['Healthy', 'PD'])
plt.ylabel('Jitter (%)')
plt.title('Jitter Comparison')
plt.show()
```

### Example 2: Handwriting Analysis
```python
import pandas as pd

df = pd.read_csv('datasets/handwriting_dataset.csv')

# Average tremor frequency by group
pd_tremor = df[df['class'] == 1]['tremor_frequency'].mean()
healthy_tremor = df[df['class'] == 0]['tremor_frequency'].mean()

print(f"PD tremor frequency: {pd_tremor:.2f} Hz")
print(f"Healthy tremor frequency: {healthy_tremor:.2f} Hz")
```

### Example 3: Clinical Progression
```python
import pandas as pd

df = pd.read_csv('datasets/ppmi_clinical_dataset.csv')
pd_data = df[df['group'] == 'PD']

# Correlation between disease duration and UPDRS
correlation = pd_data[['disease_duration', 'updrs_total']].corr()
print(correlation)
```

---

## Data Quality Notes

1. **Voice Dataset:**
   - Real data from UCI repository (or high-quality synthetic)
   - Well-validated features
   - Suitable for ML model training

2. **Handwriting Dataset:**
   - Synthetic data based on research parameters
   - Realistic feature distributions
   - Good for algorithm development

3. **PPMI Dataset:**
   - Synthetic data modeling PPMI characteristics
   - Clinically realistic ranges
   - Suitable for analysis and visualization

---

## Citations

If using these datasets for research, please cite:

**Voice Dataset:**
```
Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A., & Moroz, I. M. (2007).
Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.
BioMedical Engineering OnLine, 6(1), 23.
```

**PPMI Dataset:**
```
Parkinson Progression Marker Initiative. (2011).
The Parkinson Progression Marker Initiative (PPMI).
Progress in Neurobiology, 95(4), 629-635.
```

---

## Accessing Datasets in the Application

All datasets are automatically loaded in the application:

1. **Dataset Explorer** - Interactive visualization and exploration
2. **Voice Analysis** - Uses voice dataset for training/comparison
3. **Handwriting Analysis** - Uses handwriting dataset
4. **Model Comparison** - Evaluates models on all datasets

---

## Exporting Custom Datasets

To export filtered or processed datasets:

```python
from utils.data_loader import DataLoader

loader = DataLoader()

# Load dataset
voice_data = loader.load_voice_dataset()

# Filter and export
pd_only = voice_data[voice_data['status'] == 1]
pd_only.to_csv('pd_voice_only.csv', index=False)
```

---

## Data Privacy

- All datasets contain either public research data or synthetic data
- No personally identifiable information
- Safe for educational and research use
- Not suitable for clinical diagnosis

---

For more information on using these datasets in the application, see the main documentation and individual page descriptions.
