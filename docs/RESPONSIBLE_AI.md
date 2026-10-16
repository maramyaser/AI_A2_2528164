# Responsible AI — Musanze Harvest Intelligence Service

## 1. Intended Use

The Musanze Harvest Intelligence Service is an educational machine-learning prototype designed to demonstrate three tasks using farm-condition data:

- **Regression:** Estimate potato harvest quantity in tons.
- **Classification:** Identify farms predicted to be at high production risk.
- **Clustering:** Group farms with similar recorded conditions.

The intended stakeholders are potato cooperative field officers and students evaluating machine-learning systems. Predictions should be treated as preliminary decision-support information.

## 2. Dataset and Reproducibility

The dataset is synthetically generated using the assignment's specified generation rules and the student's registration-based random seed. It does not represent verified observations from real Musanze farms.

Results demonstrate model behavior on synthetic data only. Performance metrics cannot establish how accurately the system would predict real agricultural outcomes.

## 3. Limitations

- Synthetic records may not reflect real farming conditions, measurement errors, seasonal changes, or local agricultural practices.
- The dataset may not represent the diversity of farms, farmers, and growing conditions in Rwanda.
- Model performance depends on the generated features, selected algorithms, preprocessing, and evaluation split.
- A high classification score does not guarantee that a particular farm will experience a production problem.
- Clustering identifies statistical similarities, not definitive agricultural categories.
- The service has not been validated through field trials with farmers or agricultural specialists.

## 4. Risks and Potential Harms

Incorrect predictions could lead to inappropriate farming recommendations, unnecessary expenses, or misplaced confidence in the system. Risk classifications could unfairly influence how cooperative staff treat individual farmers if used without additional evidence.

Missing or inaccurate input values can also affect predictions. Users should verify the quality and plausibility of farm records before interpreting results.

## 5. Prohibited Uses

This prototype must not be used as the sole basis for:

- Approving or denying agricultural loans or insurance.
- Determining access to public services, grants, or other essential support.
- Penalizing farmers or excluding them from cooperative membership.
- Making binding agricultural, financial, or government decisions.
- Replacing professional agricultural assessments.

## 6. Human Oversight

A qualified agricultural professional or cooperative field officer must review relevant farm records and contextual information before acting on a prediction. Users should consider local conditions, consult the farmer when appropriate, and seek expert advice when results are uncertain.

The prediction output is not a guarantee of future yield or a diagnosis of agricultural risk.

## 7. Improvements Before Real-World Use

Before deployment, the system would require verified and representative agricultural data, consultation with farmers and agronomists, independent evaluation, monitoring for performance differences between groups and locations, and documentation of data quality. The system should be recalibrated and reassessed as farming conditions change.

## 8. Transparency

The system is an individual academic assignment. The dataset is synthetic, and the models are trained for educational evaluation. Reported metrics and conclusions must match the actual generated artifacts. Known limitations, errors, and the use of AI assistance must be disclosed honestly.
