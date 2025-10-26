# Loan Policy Optimization: Deep Learning (XGBoost) vs. Offline Reinforcement Learning (CQL)

## 🎯 Project Overview

This project implements and compares two distinct machine learning paradigms—a supervised classification model (XGBoost) and an explicit decision-making agent (Conservative Q-Learning, CQL)—to optimize loan approval policy using the **LendingClub Loan Data (2007-2018)**.

The core business objective is to **maximize the company's financial return (Expected Profit per loan)**, demonstrating the superiority of a decision-optimization framework (RL) over a traditional risk-prediction framework (DL/XGBoost).

## ✨ Key Results

The comparative analysis showed that the model explicitly optimized for the financial reward metric achieved higher profitability.

| Metric | XGBoost Model (Optimal Profit) | RL Agent (CQL) | Improvement |
| :--- | :--- | :--- | :--- |
| **Expected Profit / Loan** | **\$4,611.83** | **\$5,380.95** | **+16.67%** |
| Approval Rate | $72.30\%$ | $65.40\%$ | |
| XGBoost Validation AUC | $0.8158$ | N/A | |
| XGBoost Optimal F1-Score | $0.4398$ | N/A | |

**Conclusion:** The RL agent learns a "risk-seeking but profitable" policy, approving fewer overall loans but successfully identifying high-interest, high-value borrowers that the risk-averse XGBoost model rejects.

## 🛠️ Repository Structure (Recommended)

While the full solution was developed in a single monolithic Colab notebook for reproducibility, the final code is structured to reflect best practices:

loan-policy-optimization/ ├── README.md # This file ├── requirements.txt # Project dependencies ├── notebooks/ │ └── monolithic_solution.ipynb # The primary Colab notebook with all code └── src/ # Contains placeholder for modular code ├── preprocessing.py

├── dl_xgboost.py # Logic for XGBoost (Supervised Model) └── offline_rl.py # Logic for CQL (Decision Agent)

## ⚙️ Setup and Dependencies

This project was developed using a Google Colab environment with GPU runtime enabled for d3rlpy/PyTorch training.

### 1. Installation

```bash
# Ensure you have the necessary libraries
pip install pandas numpy scikit-learn torch torchvision torchaudio d3rlpy matplotlib seaborn xgboost tqdm

2. DataDownload the LendingClub Loan Data (specifically accepted_2007_to_2018.csv).Upload your 500,000 row stratified sample to a recognized Google Drive path (e.g., /content/drive/MyDrive/Shodh_ML_Project/df_sample_500k.csv).Update the DATA_PATH variable in the monolithic notebook's first code cell.🚀 Execution Guide (Reproducibility)The entire pipeline is designed to run sequentially within the monolithic_solution.ipynb notebook.Run Initial Setup: Execute the dependency installation and Google Drive mounting cells.Run Monolithic Code Block: Execute the main, large code block containing all function definitions and the execution flow.Key Methodology Steps:Feature Engineering & Scaling: Robust handling of missing values, creation of critical financial features (loan_to_income, credit_age_years), and scaling of the final matrix.Reward Engineering: Creation of a precise financial reward column: $r = \text{Net Interest} - \$50$ (for success) or $r = -(\text{Principal Loss} + \$50)$ (for default).Supervised Model (XGBoost): Trained using scale_pos_weight to correct class imbalance, achieving high discrimination (AUC).Threshold Optimization: An exhaustive search is performed on the XGBoost prediction probabilities to find the single optimal threshold ($\tau_{Profit}$) that maximizes the Expected Profit metric.Offline RL Agent (CQL): Trained for 50 epochs using synthetic 'deny' transitions to mitigate Survival Bias and learn a robust, profit-maximizing policy ($Q(s, a)$).Comparative Analysis: The optimal profit of the XGBoost policy is directly compared to the Estimated Policy Value of the CQL agent.📄 Final Report (Artifact)A detailed PDF report accompanies this repository, providing a deep dive into:Reward Justification: Explanation of the $\$50$ operational cost and long-term risk assessment.Metric Choice: Why Expected Policy Value is the definitive business metric.Disagreement Analysis: Specific examples where the RL agent approves a loan the XGBoost model denies, proving the value of decision optimization over pure prediction.Future Steps: Recommendations for deploying the system using A/B testing and algorithmic improvements (Hybrid RL, Safe RL).
