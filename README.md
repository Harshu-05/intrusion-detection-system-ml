# Intrusion Detection System using Machine Learning

## Objective
To design and implement an Intrusion Detection System (IDS) that detects
malicious network traffic using machine learning techniques.

## Dataset
- KDD / NSL-KDD Dataset
- Features include protocol type, service, bytes transferred, etc.
- Labels converted to:
  - 0: Normal traffic
  - 1: Attack traffic

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest Classifier

## Methodology
1. Load and preprocess network traffic data
2. Encode categorical features
3. Train Random Forest model
4. Evaluate using accuracy, confusion matrix, and classification report

## Results
- Achieved ~93% accuracy
- High precision and recall for attack detection

## Conclusion
The IDS successfully identifies malicious network activity and can be
used as a foundational security monitoring system.

## Author
Sai Harshitha Dirisala
