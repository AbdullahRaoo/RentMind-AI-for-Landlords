# AI Tenant Screening Logic
# Simple rule-based model for initial version

def screen_tenant(credit_score, income, rent, employment_status, eviction_record):
    """
    Screen a tenant application using simple rule-based criteria.

    Criteria:
    - Minimum credit score: 600 (below this is high risk)
    - Income-to-rent ratio: Income must be at least 3x the monthly rent
    - Employment status: 'employed' is preferred; others add risk
    - Eviction record: Any prior eviction is a major risk

    Returns:
        dict with:
            - risk_score: int (higher means riskier)
            - recommendation: str ("Accept", "Review", or "Reject")
            - explanation: str (human-readable explanation of the decision)
    Args:
        credit_score (int): Applicant's credit score
        income (float): Applicant's monthly income
        rent (float): Monthly rent for the property
        employment_status (str): Employment status (e.g., 'employed', 'unemployed')
        eviction_record (bool): True if applicant has prior eviction, else False
    """
    explanation = []
    risk_score = 0
    # Credit score rule
    if credit_score < 600:
        explanation.append(f"Credit score {credit_score} is below minimum (600).")
        risk_score += 40
    else:
        explanation.append(f"Credit score {credit_score} meets minimum.")
    # Income-to-rent rule
    if income < 3 * rent:
        explanation.append(f"Income (£{income}) is less than 3x rent (£{rent}).")
        risk_score += 40
    else:
        explanation.append(f"Income (£{income}) is at least 3x rent (£{rent}).")
    # Employment status
    if employment_status.lower() != "employed":
        explanation.append(f"Employment status is '{employment_status}'.")
        risk_score += 10
    else:
        explanation.append("Employment status is employed.")
    # Eviction record
    if eviction_record:
        explanation.append("Prior eviction record found.")
        risk_score += 30
    else:
        explanation.append("No prior eviction record.")
    # Recommendation
    if risk_score >= 60 or credit_score < 500 or eviction_record:
        recommendation = "Reject"
    elif risk_score >= 30:
        recommendation = "Review"
    else:
        recommendation = "Accept"
    return {
        "risk_score": risk_score,
        "recommendation": recommendation,
        "explanation": "\n".join(explanation)
    }
