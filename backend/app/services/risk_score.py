def calculate_risk_score(
    meta_report: dict,
    ocr_report: dict,
    ml_prediction: dict = None,
    ocr_failed: bool = False,
    missing_fields: bool = False,
    validation_mismatch: bool = False,
<<<<<<< HEAD
    graph_risk_penalty: int = 0,
    graph_reason: str = "",
    possible_forgery: bool = False,
    signature_similarity: float = 1.0,
    gnn_fraud_probability: float = 0.0,
    gnn_risk_level: str = "Low"
=======
    ela_score: float = 0.0,
    compress_report: dict = None,
    quality_report: dict = None
>>>>>>> 930bc4133369cb20a5ebd19be27a311f92dc5b81
) -> dict:
    """
    Computes a fraud risk score (0-100) and risk level classification:
    - AI/ML ResNet18 Classifier penalty (up to +35 based on confidence)
    - Metadata anomaly = +30
    - OCR extraction failure = +20
    - Missing required fields = +25
    - Validation mismatch = +25
<<<<<<< HEAD
    - Graph syndicate fraud ring penalty (up to +30)
=======
    - ELA discrepancy = up to +25
    - Compression artifacts = +15
    - Image quality warning = +10
>>>>>>> 930bc4133369cb20a5ebd19be27a311f92dc5b81
    """
    score = 0
    issues = []

    # 0. AI/ML ResNet18 prediction penalty
    if ml_prediction and ml_prediction.get("prediction") == "tampered":
        confidence = ml_prediction.get("confidence", 0.5)
        penalty = int(35 * confidence)
        score += penalty
        issues.append(f"AI/ML ResNet18 Classifier flagged the document as tampered (confidence: {confidence * 100:.1f}%).")

    # 1. Metadata anomaly = +30
    is_meta_anomaly = (
        meta_report.get("status") in ["Alert", "Tampered"] or 
        len(meta_report.get("warnings", [])) > 0
    )
    if is_meta_anomaly:
        score += 30
        for warning in meta_report.get("warnings", []):
            issues.append(f"Metadata check: {warning}")
        if not meta_report.get("warnings"):
            issues.append("Metadata anomaly detected (editing software or timestamp alteration flags).")

    # 2. OCR extraction failure = +20
    if ocr_failed:
        score += 20
        issues.append("OCR text extraction failure.")
    else:
        text_blocks = ocr_report.get("text_blocks", [])
        if not text_blocks:
            score += 20
            issues.append("OCR text extraction failure (no text blocks detected).")
            ocr_failed = True

    # 3. Missing required fields = +25
    if not ocr_failed:
        text_str = " ".join([b.get("text", "") for b in ocr_report.get("text_blocks", [])]).lower()
        required_fields = ["applicant name", "property address", "monthly income", "loan amount"]
        missing_any = any(field not in text_str for field in required_fields)
        if missing_any:
            score += 25
            issues.append("Missing required fields (e.g. Applicant Name, Income, or Property details).")
    else:
        score += 25
        issues.append("Missing required fields due to OCR failure.")

    # 4. Validation mismatch = +25
    has_font_variance = ocr_report.get("font_analysis", {}).get("status") == "Alert"
    has_sig_anomaly = ocr_report.get("signature_analysis", {}).get("status") == "Alert"
    if has_font_variance or has_sig_anomaly or validation_mismatch:
        score += 25
        issues.append("Validation mismatch or font/signature authenticity layout variance.")

<<<<<<< HEAD
    # 5. Graph syndicate penalty
    if graph_risk_penalty > 0:
        score += graph_risk_penalty
        issues.append(f"Syndicate network alert: {graph_reason}")

    # 6. Signature forgery penalty
    if possible_forgery:
        score += 30
        issues.append(f"Possible signature forgery detected. Signature similarity score of {signature_similarity * 100:.1f}% is below threshold.")

    # 7. GNN syndicate fraud detection penalty
    if gnn_fraud_probability >= 0.5:
        gnn_penalty = int(30 * gnn_fraud_probability)
        score += gnn_penalty
        issues.append(f"GNN Graph Convolutional Network flagged high syndicate fraud risk: {gnn_fraud_probability * 100:.1f}% probability ({gnn_risk_level} risk level).")
=======
    # 5. ELA score = up to +25
    if ela_score > 35.0:
        added = min(int((ela_score - 35) * 0.5) + 10, 25)
        score += added
        issues.append(f"Error Level Analysis (ELA) discrepancy detected (Score: {ela_score}%).")

    # 6. Compression warnings = +15
    if compress_report and compress_report.get("status") == "Alert":
        score += 15
        for warning in compress_report.get("warnings", []):
            issues.append(f"Compression check: {warning}")

    # 7. Image quality warnings = +10
    if quality_report and quality_report.get("status") == "Alert":
        score += 10
        for warning in quality_report.get("warnings", []):
            issues.append(f"Image quality check: {warning}")
>>>>>>> 930bc4133369cb20a5ebd19be27a311f92dc5b81

    # Bound risk score between 0 and 100
    score = min(max(score, 0), 100)

    # Risk level classification
    if score <= 35:
        risk_level = "Low"
    elif score <= 65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "issues": issues
    }
