# fusion_module.py

# Weights based on F1-score proportional fusion
ALPHA = 0.5316  # Questionnaire weight
BETA = 0.4684   # Eye-gaze weight


def fuse_scores(Q_score, G_score):
    """
    Combines questionnaire and eye-gaze probabilities
    using weighted decision-level fusion.

    Parameters:
        Q_score (float): Probability from questionnaire model (0 to 1)
        G_score (float): Probability from eye-gaze model (0 to 1)

    Returns:
        final_score (float)
        prediction (str)
        risk_level (str)
    """

    # Safety check
    if not (0 <= Q_score <= 1 and 0 <= G_score <= 1):
        raise ValueError("Scores must be between 0 and 1")

    # Weighted fusion
    final_score = ALPHA * Q_score + BETA * G_score

    # Final classification
    if final_score >= 0.5:
        prediction = "ASD Risk Detected"
    else:
        prediction = "Typical Development Pattern"

    # Risk level interpretation
    if final_score >= 0.75:
        risk_level = "High Risk"
    elif final_score >= 0.5:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return final_score, prediction, risk_level