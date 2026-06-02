def calculate_risk_score(
    meta_report: dict,
    ocr_report: dict,
    ocr_failed: bool = False,
    missing_fields: bool = False,
    validation_mismatch: bool = False
) -> dict:
    """
    Computes a fraud risk score (0-100) and risk level classification:
    - Metadata anomaly = +30
    - OCR extraction failure = +20
    - Missing required fields = +25
    - Validation mismatch = +25
    """
    score = 0
    issues = []

    # 1. Metadata anomaly = +30
    is_meta_anomaly = (
        meta_report.get("status") in ["Alert", "Tampered"] or 
        len(meta_report.get("warnings", [])) > 0
    )
    if is_meta_anomaly:
        score += 30
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
