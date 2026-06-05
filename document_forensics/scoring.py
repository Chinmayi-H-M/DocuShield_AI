def calculate_authenticity_score(suspicious_regions_count: int, metadata_analysis: dict) -> tuple:
    """
    Computes the document authenticity score out of 100 based on refined hackathon rules.
    
    Deductions:
    - Suspicious ELA regions (images): -35 points
    - Font Consistency Warnings (experimental): -5 points
    - Editing software footprint (Photoshop, Canva, GIMP): -10 points
    - PDF Incremental updates (multiple EOF saves): -15 points
    - Temporal date anomalies (post-creation edit or spoofing): -15 points
    - Missing standard fields (stripped EXIF or empty PDF fields): -10 points
    - Combination Penalty (software detected AND ELA/Incremental saves found): -20 points
    
    Parameters:
        suspicious_regions_count (int): Number of suspicious image regions detected by OpenCV.
        metadata_analysis (dict): The output dictionary from analyze_metadata_anomalies in metadata.py.
        
    Returns:
        tuple: (score, deductions_list)
               - score (int): Final authenticity score bounded between 0 and 100.
               - deductions_list (list): List of dicts specifying {"factor": str, "points": int}.
    """
    score = 100
    deductions = []
    
    # 1. Deduct for suspicious ELA regions
    if suspicious_regions_count > 0:
        points_deducted = 35
        deductions.append({
            "factor": f"Suspicious ELA regions detected ({suspicious_regions_count} regions found)",
            "points": points_deducted
        })
        score -= points_deducted
        
    # 2. Deduct for Font Consistency Warnings (Experimental)
    font_warnings = metadata_analysis.get("font_consistency_warnings", [])
    if font_warnings:
        points_deducted = 5
        deductions.append({
            "factor": f"Font consistency warnings detected ({len(font_warnings)} warnings)",
            "points": points_deducted
        })
        score -= points_deducted
        
    # 3. Deduct for editing software signatures
    if metadata_analysis.get("editing_software_detected", False):
        points_deducted = 10
        deductions.append({
            "factor": f"Editing software metadata signature found ({metadata_analysis.get('software_signature')})",
            "points": points_deducted
        })
        score -= points_deducted
        
    # 4. Deduct for PDF Incremental Updates (multiple EOF saves)
    if metadata_analysis.get("has_pdf_structural_warnings", False):
        points_deducted = 15
        deductions.append({
            "factor": f"PDF contains incremental updates ({metadata_analysis.get('incremental_updates_count')} saves detected)",
            "points": points_deducted
        })
        score -= points_deducted
        
    # 5. Deduct for date/temporal anomalies
    has_date_anomaly = any("temporal" in anomaly.lower() or "date" in anomaly.lower() 
                           for anomaly in metadata_analysis.get("anomalies", []))
    if has_date_anomaly:
        points_deducted = 15
        deductions.append({
            "factor": "Temporal date anomalies detected (e.g. document modified post-creation or date spoofing)",
            "points": points_deducted
        })
        score -= points_deducted
        
    # 6. Deduct for missing standard fields or stripped EXIF
    has_missing_fields = any("missing" in anomaly.lower() or "stripped" in anomaly.lower() 
                             for anomaly in metadata_analysis.get("anomalies", []))
    if has_missing_fields:
        points_deducted = 10
        deductions.append({
            "factor": "Standard document metadata fields are missing or stripped",
            "points": points_deducted
        })
        score -= points_deducted
        
    # 7. Combined Combo Penalty
    # Triggered when editing software is detected AND (ELA contours found OR PDF has incremental saves)
    has_modification_proof = (suspicious_regions_count > 0) or metadata_analysis.get("has_pdf_structural_warnings", False)
    if metadata_analysis.get("editing_software_detected", False) and has_modification_proof:
        points_deducted = 20
        deductions.append({
            "factor": "Combination Penalty: Editing software signature coupled with active content changes",
            "points": points_deducted
        })
        score -= points_deducted
        
    # Ensure score does not fall below 0
    score = max(0, score)
    
    return score, deductions

def classify_risk(score: int) -> str:
    """
    Classifies the document risk category based on the final Authenticity Score.
    
    Risk Ranges:
    - 90 to 100: Genuine / Low Risk
    - 50 to 89: Medium Risk
    - 0 to 49: High Risk
    
    Parameters:
        score (int): The authenticity score (0-100).
        
    Returns:
        str: The risk classification label.
    """
    if score >= 90:
        return "Genuine / Low Risk"
    elif score >= 50:
        return "Medium Risk"
    else:
        return "High Risk"

def generate_explanation(score: int, risk_level: str, deductions: list) -> str:
    """
    Generates a clear natural language explanation of the scoring and risk level.
    This explains the AI forensics results transparently.
    
    Parameters:
        score (int): The final calculated score.
        risk_level (str): The risk classification string.
        deductions (list): The list of applied deductions.
        
    Returns:
        str: A multi-line natural language explanation.
    """
    if score == 100:
        return (
            "The document is classified as Genuine / Low Risk. No metadata anomalies, "
            "missing fields, or suspicious compression variances were detected. "
            "The document appears untampered."
        )
        
    explanation_parts = [
        f"The document is classified as {risk_level} with an Authenticity Score of {score}/100."
    ]
    
    if deductions:
        explanation_parts.append("Deductions were applied for the following reasons:")
        for dec in deductions:
            explanation_parts.append(f"  - {dec['factor']} (-{dec['points']} pts)")
            
    # Add a friendly context summary
    if risk_level == "Genuine / Low Risk":
        explanation_parts.append(
            "\nNote: Minor warnings or missing fields were noted, but they do not suggest active forgery. "
            "The document is likely authentic."
        )
    elif risk_level == "Medium Risk":
        explanation_parts.append(
            "\nWarning: The document contains warnings (e.g. edited with Canva or has a date gap), "
            "but lacks strong proof of graphic alterations (no ELA anomalies or multiple saves found). "
            "Underwriting manual review is recommended."
        )
    elif risk_level == "High Risk":
        explanation_parts.append(
            "\nCritical Alert: High risk of tampering. Multiple critical anomalies exist. "
            "We detected active graphic alterations (ELA regions) or structural changes (incremental saves) "
            "combined with editing tool footprints or date inconsistencies."
        )
        
    return "\n".join(explanation_parts)
