# Credit Card Fraud Detection using Genetic Algorithm

This project implements a genetic algorithm (GA) to detect fraudulent credit card transactions. The algorithm optimizes a set of thresholds for selected features to classify transactions as fraudulent or non-fraudulent, using the F1-score as the fitness function. The project achieves an F1-score of 0.7009 on the test set, with a precision of 0.6048 and recall of 0.8333.

## Project Overview

The goal of this project is to develop a fraud detection system using a genetic algorithm. The algorithm processes a dataset of credit card transactions, balancing the classes with SMOTE and optimizing feature thresholds to maximize the F1-score. The implementation includes data preprocessing, genetic algorithm optimization, and evaluation on a test set.

### Key Features
- **Dataset**: Credit card transaction data with features like transaction amount, declines, and chargeback history.
- **Preprocessing**: Encoding categorical variables, filling missing values, and normalizing numerical features.
- **Class Balancing**: SMOTE is used to address class imbalance (85.4% non-fraudulent vs. 14.6% fraudulent transactions).
- **Genetic Algorithm**: Optimizes thresholds for 9 features, using a fixed classification rule (4 or more features exceeding thresholds indicate fraud).
- **Evaluation**: Precision, recall, and F1-score are computed on the test set.
- **Visualization**: A plot of the best F1-score over generations is generated (`fitness_over_generations.png`).

## Dataset

The dataset used in this project is sourced from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud). It contains 3075 transactions with 12 features, including:
- Numerical features: `Average Amount/transaction/day`, `Transaction_amount`, `Total Number of declines/day`, `Daily_chargeback_avg_amt`, `6_month_avg_chbk_amt`, `6-month_chbk_freq`.
- Categorical features: `Is declined`, `isForeignTransaction`, `isHighRiskCountry`, `isFradulent` (target).
- Other features: `Merchant_id`, `Transaction date` (not used in the model).

The dataset is imbalanced, with 2627 non-fraudulent (85.4%) and 448 fraudulent (14.6%) transactions.

## Requirements

To run this project, you need Python 3.6+ and the following libraries:
- pandas
- numpy
- scikit-learn
- imblearn
- matplotlib

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn imblearn matplotlib
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection-ga.git
   cd credit-card-fraud-detection-ga
   ```

2. **Download the dataset**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).
   - Place the `Credit Card Fraud Detection Data.csv` file in the project directory.

3. **Set up the environment**:
   Ensure all required libraries are installed (see Requirements).

## Usage

1. **Open the Jupyter Notebook**:
   Launch Jupyter Notebook and open `GenAlg.ipynb`:
   ```bash
   jupyter notebook GenAlg.ipynb
   ```

2. **Update the file path**:
   In the second cell of the notebook, set the `file_path` to the location of `Credit Card Fraud Detection Data.csv`:
   ```python
   file_path = 'Credit Card Fraud Detection Data.csv'
   ```

3. **Run the notebook**:
   Execute all cells in the notebook. The script will:
   - Load and preprocess the dataset.
   - Apply SMOTE to balance the classes.
   - Run the genetic algorithm for 100 generations.
   - Evaluate the model on the test set and print precision, recall, and F1-score.
   - Save a plot of the fitness history (`fitness_over_generations.png`).

### Expected Output
The notebook will output:
- Dataset size and class distribution.
- Training and test set sizes.
- F1-score for each generation (e.g., reaching 0.8705 by generation 100).
- Test set metrics (e.g., Precision: 0.6048, Recall: 0.8333, F1-score: 0.7009).
- A plot showing the improvement of the F1-score over generations.

## Genetic Algorithm Details

- **Population Size**: 100 individuals.
- **Mutation Rate**: 0.2, decreasing linearly over generations.
- **Generations**: 100.
- **Elite Size**: 5 (top individuals preserved each generation).
- **Fitness Function**: F1-score, computed by classifying transactions based on whether 4 or more features exceed their respective thresholds.
- **Selection**: Tournament selection (size 5).
- **Crossover**: Single-point crossover.
- **Mutation**: Randomly resets thresholds within [-3, 3] for normalized features.

## Results

The genetic algorithm achieves:
- **Training F1-score**: Up to 0.8705 after 100 generations.
- **Test F1-score**: 0.7009, with a precision of 0.6048 and recall of 0.8333.

The high recall (0.8333) indicates the model is effective at identifying fraudulent transactions, though the moderate precision (0.6048) suggests some false positives.

## Future Improvements

- **Threshold Tuning**: Experiment with different classification thresholds (e.g., 3 or 5 instead of 4).
- **Feature Weighting**: Introduce weights for features to prioritize more predictive ones.
- **Alternative Metrics**: Use ROC-AUC or weighted F1-score for fitness evaluation.
- **Feature Selection**: Analyze feature correlations to exclude less informative features.
- **Hyperparameter Tuning**: Adjust population size, mutation rate, or number of generations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is provided by [Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud).
- Built using Python, scikit-learn, and imblearn libraries.