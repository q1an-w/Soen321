Credit Card Fraud Detection Security Tests: An Analysis of Machine Learning Model Vulnerabilities

Authors:
Karina Sanchez-Duran (40189860)
Qian Yi Wang (40211303)
Paul Humennyj (40209588)
Vanessa DiPietrantonio (40189938)
Mohamad Mounir Yassin (40198854)
Yash Patel (40175454)

# Page 1: Key Findings Summary
## Major Findings
- [Key finding 1]
- [Key finding 2]
- [Key finding 3]

## Security Implications
- [Security implication 1]
- [Security implication 2]
- [Security implication 3]

## Recommendations
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

# Pages 2-4: Methodologies and Tools
## 1. Base Model Implementation
### 1.1 Dataset Description
### 1.2 Model Architecture
### 1.3 Training Methodology
### 1.4 Baseline Performance Metrics

## 2. Attack Implementations
### 2.1 Adversarial Attacks
#### 2.1.1 White-box Attacks
#### 2.1.2 Black-box Attacks
### 2.2 Evasion Attacks
In order to check the robustness of the trained fraud detection model, we implemented multiple evasionattacks based on input generated by a surrogate (black-box) model. These attack make a scenario where an attacker, having no acces to the internal model, crafts inputs to evade fraud detection at test time.

Each evasion attack slightly perturbs known fraudulent inputs and tests whether the modified inputs are misclassified as non-fraud by the original decision tree classifier. The goal is to determine how easily fraud can bypass detection with minimal manipulation.

We structured our attacks using separate strategy scripts to preserve clarity and modular testing.

#### Strategy A:
This strategy reduced the values of `feature5` and `feature7` by 1, assuming these features contribute significantly to fraud detection. After applying this transformation and testing against the original model, we achieved:
- **Evasion Success**: 35 out of 226 samples
- **Success Rate**: **15.49%**
The result shows that even small input modifications to key features can lead to incorrect non-fraud classification by the model.

#### Strategy B:
This strategy incremented the values of `feature3` and `feature6` by 1 for each fraudulent input. These changes were hypothesized to make fraudulent behavior appear more benign.

- **Evasion Success**: 226 out of 226 samples
- **Success Rate**: **100.00%**

This strategy was completely successful. The model misclassified every modified fraudulent input as legitimate, highlighting a critical vulnerability. Even slight increases in these two features significantly impacted the model's decision boundary.

#### Strategy C:
This strategy randomly perturbed every feature (either +1 or -1) for each fraudulent record. It simulates a real-world scenario where an attacker modifies multiple features slightly to evade detection without knowing which ones matter.

- **Evasion Success:** 169 out of 226 samples
- **Success Rate**: **74.78%**

The high success rate highlights the model’s vulnerability to widespread but subtle feature changes. Even without targeting specific features, nearly 75% of the modified fraudulent inputs were misclassified as legitimate. While not as precise as Strategy B, this approach still bypassed the model in the majority of cases. It suggests that the model's decision boundary is highly sensitive to broad feature noise, not just targeted changes.

### 2.3 Membership Inference Attacks

## 3. Testing Framework
### 3.1 Performance Testing Setup
### 3.2 Security Testing Methodology
### 3.3 Tools and Libraries Used
#### 3.3.1 Foolbox Implementation
#### 3.3.2 Locust Testing Setup
#### 3.3.3 PyTorch Adversarial Attack Tools

# Pages 5-7: Results
## 1. Model Performance Analysis
### 1.1 Baseline Model Performance
### 1.2 Performance Under Attack

## 2. Attack Effectiveness Analysis
### 2.1 Adversarial Attack Results
#### 2.1.1 White-box Attack Success Rates
#### 2.1.2 Black-box Attack Success Rates
### 2.2 Evasion Attack Results
### 2.3 Membership Inference Attack Results

## 3. Comparative Analysis
### 3.1 Attack Method Comparison
### 3.2 Resource Consumption Analysis
### 3.3 Detection Rate Analysis

# Pages 8-9: Discussion and Practical Implications
## 1. Security Vulnerability Analysis
### 1.1 Critical Vulnerabilities Identified
### 1.2 Risk Assessment

## 2. Defense Strategies
### 2.1 Proposed Countermeasures
### 2.2 Implementation Recommendations
### 2.3 Trade-offs and Limitations

## 3. Real-world Applications
### 3.1 Industry Impact
### 3.2 Implementation Considerations
### 3.3 Future Research Directions

# Page 10: References
## Tools and Libraries
## Academic References
## Dataset Sources

# Appendices
## Appendix A: Detailed Attack Results

### Strategy A
A selection of the modified fraudulent inputs and their outcomes:

| Sample | Original Feature5 | Modified Feature5 | Original Feature7 | Modified Feature7 | Prediction | Evasion Success |
|--------|-------------------|-------------------|-------------------|-------------------|------------|------------------|
| 1      | 5                 | 4                 | 5                 | 4                 | 1          | NO               |
| 4      | 4                 | 3                 | 1                 | 0                 | 0          | YES              |
| 20     | 1                 | 0                 | 1                 | 0                 | 0          | YES              |
| 47     | 3                 | 2                 | 1                 | 0                 | 0          | YES              |
| 99     | 0                 | 0                 | 2                 | 1                 | 0          | YES              |

*Full result CSV available as `results_strategy_a.csv` in the Evasion_Strategies folder.*

### Strategy B
A selection of modified fraudulent inputs and their outcomes:

| Sample | Original Feature3 | Modified Feature3 | Original Feature6 | Modified Feature6 | Prediction | Evasion Success |
|--------|-------------------|-------------------|-------------------|-------------------|------------|------------------|
| 1      | 5                 | 6                 | 0                 | 1                 | 0          | YES              |
| 5      | 5                 | 6                 | 0                 | 1                 | 0          | YES              |
| 17     | 5                 | 6                 | 0                 | 1                 | 0          | YES              |
| 32     | 5                 | 6                 | 0                 | 1                 | 0          | YES              |
| 100    | 5                 | 6                 | 0                 | 1                 | 0          | YES              |

*Full CSV output saved as `results_strategy_b.csv`.*

#### Strategy C
A selection of randomly modified fraudulent inputs and their outcomes:

| Sample | Feature Changed | Original Value | Modified Value | Prediction | Evasion Success |
|--------|------------------|----------------|----------------|------------|------------------|
| 1      | feature2         | 0              | 1              | 1          | NO               |
| 2      | feature6         | 0              | 1              | 0          | YES              |
| 9      | feature7         | 3              | 4              | 0          | YES              |
| 21     | feature5         | 1              | 0              | 0          | YES              |
| 46     | feature6         | 0              | 1              | 0          | YES              |

*Full result CSV available as `results_strategy_c.csv` in the Evasion_Strategies folder.*

## Appendix B: Code Snippets
## Appendix C: Additional Performance Metrics
## Appendix D: Test Cases
## Appendix E: Implementation Details
