iFood Marketing Campaign Response - Random Forest Prediction
This project predicts customer response to marketing campaigns using Random Forest classification with cost-sensitive threshold optimization. The analysis includes feature engineering, hyperparameter tuning via GridSearchCV, and business-driven decision thresholds to minimize marketing costs.

Project Structure
ifood marketing response/
├── ifood_df.csv                          # Dataset
├── ifood marketing response.py                 # Main script (run this)
├── results/                              # Generated outputs
│   ├── feature_importance.png
│   ├── confusion_matrix_050.png
│   ├── confusion_matrix_cost_optimal.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── performance_metrics.png
│   ├── feature_analysis.png
│   └── model_summary.txt
└── README.md                             # This file

**Data**
Raw Data: The dataset (ifood_df.csv) contains customer demographics, purchasing behavior, and campaign response history. Target variable Response indicates whether a customer responded to the marketing campaign (1 = Yes, 0 = No). Response rate: 15.10%.

**Install Dependencies**
Python 3.x

pandas — Data manipulation

numpy — Numerical computing

scikit-learn — Machine learning (RandomForestClassifier, GridSearchCV)

matplotlib — Visualization

seaborn — Statistical visualization

Run the Main Script
bash
python ifood marketing response.py
Actual Results from a Recent Run
See Figures 1 to 8

**Model**:

Random Forest classifier (200 trees) with balanced class weights and cost-sensitive threshold optimization.

Performance metrics (threshold 0.50):

Cross-validation ROC-AUC: 0.8931

Test ROC-AUC: 0.8834

Test Accuracy: 0.8753

Average Precision: 0.5736

Best hyperparameters:

n_estimators: 200, max_depth: None, max_features: sqrt, min_samples_split: 5, min_samples_leaf: 2

Cost-sensitive optimization:

Optimal threshold: 0.10 (vs. default 0.50)

Minimum expected cost: $61,300

Cost assumptions: Missing a responder (FN) = $5,000, Contacting non-responder (FP) = $300

Confusion matrix comparison:

Threshold 0.50: 43 false negatives, 12 false positives (misses 64% of responders)

Threshold 0.10: 2 false negatives, 171 false positives (captures 97% of responders)

**Generated figures**:

Figure 1: Top 20 feature importances

Figure 2-3: Confusion matrices (default & cost-optimal thresholds)

Figure 4: ROC curve (AUC = 0.883)

Figure 5: Precision-Recall curve (AP = 0.574)

Figure 6: Performance metrics bar chart

Figure 7: Feature analysis by response status

Figure 8: Summary report (text file)

**Plain-Language Explanation for Non-Experts**:

The model identifies which customers will likely respond to marketing campaigns. Instead of using the standard 0.50 threshold, we use 0.10 because missing a responder costs $5,000 (lost opportunity) while contacting a non-responder only costs $300 (marketing expense).

Key insight: At threshold 0.10, we contact more customers (171 extra non-responders) but only miss 2 potential responders instead of 43. This aggressive strategy minimizes total costs because false negatives are 17× more expensive than false positives.

Business impact: The cost-optimized approach captures 97% of responders (65 out of 67) while keeping expected costs at $61,300—significantly lower than the $215,000+ from using the default threshold.

Important Considerations
Cost assumptions: Adjust the $5,000 FN and $300 FP costs based on your actual campaign economics


**License**
This project is licensed under the MIT License.

You are FREE to:

✅ Use for personal or commercial projects

✅ Modify and adapt the code

✅ Share or redistribute

Just keep the original copyright notice.

## Acknowledgments

- Dataset: [Amrah Ismayilzada](https://www.kaggle.com/code/amrahismayilzada/marketing-data-random-forest/input) on Kaggle