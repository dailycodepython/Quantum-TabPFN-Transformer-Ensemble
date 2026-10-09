# Quantum-TabPFN-Transformer-Ensemble 🚀

[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.10+](https://shields.io)](https://python.org)
[![GPU Acceleration](https://shields.io)](https://nvidia.com)

A hybrid quantum-classical ensemble learning framework that combines **Quantum Machine Learning (QML)** variational principles, **Complex Variables (CVQT)** feature engineering, and the **TabPFN Transformer** foundation model alongside state-of-the-art gradient boosting algorithms (XGBoost, LightGBM, CatBoost) for highly efficient tabular data classification.

---

## 📐 Theoretical Framework & Feature Engineering

The core pipeline bypasses traditional tabular limitations by shifting classical attributes into a **Quantum-Complex Field Space**, extracting high-dimensional patterns before model ingestion:

1. **Vector Discomfort Index (Linear Algebra):** Aggregates dimensional error vectors using spatial Euclidean norms to track physical flight boundaries.
2. **Complex Variable Analysis (CVQT):** Maps time-domain delay dynamics into the complex plane (\(z = x + iy\)) to evaluate phase shifts (\(\text{Deg}\)) and operational magnitude forces.
3. **Bloch Sphere Quantum Mapping:** Encodes normalized service delta ratings and complex stress features into quantum state parameters (\(\phi, \theta\)). The exact state amplitude \(\alpha\) is derived, and Born's Rule is applied to capture non-linear interactions via pure quantum projection:
   \[P_{\text{satisfied}} = \vert{}\alpha\vert{}^2\]

---

## 🏗️ Architecture Blueprint

The following master diagram details the unified end-to-end data processing, feature transformation, subsampling guardrails, and parallel ensemble meta-blending architecture:

```mermaid
graph TD
    %% Node Styling
    classDef raw fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px;
    classDef engineering fill:#efe5fd,stroke:#7e57c2,stroke-width:2px;
    classDef quantum fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef split fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef model fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;

    %% STAGE 1: Data Inputs
    subgraph Stage_1 [1. Data Input & Ingestion]
        DF[train.csv / test.csv]:::raw
        Delays[Departure & Arrival Delays]:::raw
        Services[14 Customer Service Ratings]:::raw
        Distance[Flight Distance]:::raw
        
        DF --> Delays & Services & Distance
    end

    %% STAGE 2: Quantum-Complex Feature Engineering
    subgraph Stage_2 [2. Quantum-Complex Feature Engineering]
        %% Linear Algebra
        V_Dist["v_error_dist = sqrt(dep_delay² + arr_delay²)"]
        V_Index["Vector_Discomfort_Index"]
        Distance & Delays --> V_Dist --> V_Index
        
        %% Complex Analysis
        Z_Array["Complex State:<br>z_array = x + i*y"]
        Stress_Mag["Complex_Stress_Magnitude"]
        Stress_Phase["Stress_Phase_Deg"]
        Delays --> Z_Array --> Stress_Mag & Stress_Phase
        
        %% Quantum Mapping & Born's Rule
        Scaler[StandardScaler]
        Phi["Phi Angle<br>(Service Delta)"]
        Theta["Theta Angle<br>(Stress Magnitude)"]
        Alpha["Quantum Amplitude<br>alpha = f(phi, theta)"]:::quantum
        Q_Prob["Quantum_Prob_Satisfied<br>|alpha|²"]:::quantum
        
        Services --> Scaler --> Phi
        Stress_Mag --> Theta
        Phi & Theta --> Alpha --> Q_Prob
    end

    %% STAGE 3: Dataset Assembly & Validation Split
    subgraph Stage_3 [3. Dataset Assembly & Validation Strategy]
        Features["Final Feature Matrix X<br>[Base + Quantum-Complex + Service]"]
        Target["Target y (LabelEncoded)"]
        Split["Stratified Train/Val Split (80/20)"]:::split
        
        V_Index & Stress_Phase & Q_Prob --> Features
        Features & Target --> Split
        
        X_train_full["Full X_train"]:::split
        X_val["X_val"]:::split
        
        Split --> X_train_full & X_val
    end

    %% STAGE 4: Parallel Ensemble Training
    subgraph Stage_4 [4. Parallel Ensemble Training]
        %% Classical Boosting Branch
        XGB["XGBoost Classifier<br>(max_depth=7)"]:::model
        LGB["LightGBM Classifier<br>(max_depth=8)"]:::model
        Cat["CatBoost Classifier<br>(depth=7)"]:::model
        
        X_train_full --> XGB & LGB & Cat
        X_val --> XGB & LGB & Cat
        
        %% TabPFN Foundation Branch
        Subsample["Subsampling Context<br>(Random 10,000 rows)"]:::split
        X_train_sub["X_train_sub"]:::split
        TabPFN["TabPFN Classifier<br>(GPU In-Context Transformer)"]:::model
        
        X_train_full --> Subsample --> X_train_sub
        X_train_sub & X_val --> TabPFN
    end

    %% STAGE 5: Meta-Blending & Prediction Output
    subgraph Stage_5 [5. Meta-Blending & Predictions]
        Blend{"Weighted Ensemble Blending<br>0.35*XGB + 0.25*LGB + 0.25*Cat + 0.15*TabPFN"}:::engineering
        
        XGB & LGB & Cat & TabPFN --> |predict_proba| Blend
        
        Metric["Validation Score<br>roc_auc_score(y_val, blend)"]:::output
        Submission["Kaggle Submission DataFrame"]:::output
        CSV["submission.csv"]:::output
        
        Blend --> Metric & Submission --> CSV
    end
```

---

## 🛠️ Installation & Setup

Clone the repository and install the verified computational dependency tree:

```bash
git clone https://github.com
cd Quantum-TabPFN-Transformer-Ensemble
pip install -r requirements.txt
```

### Core Requirements
* `python >= 3.10`
* `torch >= 2.0` (CUDA GPU support strongly recommended for efficient TabPFN inference)
* `tabpfn`
* `xgboost`, `lightgbm`, `catboost`
* `pandas`, `numpy`, `scikit-learn`

---

## 🚀 Quick Start / Usage

Execute the pipeline to perform feature transformations, parallel training, and generate the meta-ensemble submissions:

```python
from ensemble_pipeline import apply_advanced_math_features, run_meta_ensemble
import pandas as pd

# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Trigger the Quantum-Complex Field Engine
train_enriched = apply_advanced_math_features(train_df)
test_enriched = apply_advanced_math_features(test_df)

# 3. Fit models and perform optimized ensemble meta-blending
# Blending distribution weights: 35% XGB / 25% LGBM / 25% CatBoost / 15% TabPFN
submission = run_meta_ensemble(train_enriched, test_enriched)
submission.to_csv('submission.csv', index=False)
print("Pipeline executed successfully. Meta-ensemble submission.csv is ready!")
```

---

## 📊 Meta-Blending Distribution Matrix

| Classifier Model | Model Strategy Type | Assigned Blending Weight | Key Optimization Parameter |
| :--- | :--- | :---: | :--- |
| **XGBoost** | Gradient Tree Boosting | **35%** | `max_depth=7`, `eval_metric='logloss'` |
| **LightGBM** | Leaf-wise Tree Growth | **25%** | `max_depth=8`, Fast GPU routing |
| **CatBoost** | Symmetric Deep Trees | **25%** | `depth=7`, `l2_leaf_reg=5.0` |
| **TabPFN** | Tabular In-Context Transformer | **15%** | GPU Acceleration, `size=10000` context window |

---

## 📄 License
Distributed under the **MIT License**. Read `LICENSE` for more information.


