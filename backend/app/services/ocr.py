import os

def analyze_ocr_layout(file_path: str, original_filename: str = None) -> dict:
    """
    Simulates LayoutLMv3 and PaddleOCR bounding box extraction.
    Returns extracted labels, positions, confidence levels, and font consistency statuses.
    """
    basename = (original_filename if original_filename else os.path.basename(file_path)).lower()
    
    # Setup standard text and layout boxes
    layout_data = {
        "text_blocks": [
            {"text": "CANARA BANK LOAN APPLICATION", "x": 100, "y": 80, "width": 400, "height": 30, "confidence": 99.4},
            {"text": "Applicant Name: Ramesh Kumar", "x": 80, "y": 180, "width": 250, "height": 20, "confidence": 98.9},
            {"text": "Co-Applicant: Sunita Kumar", "x": 80, "y": 210, "width": 250, "height": 20, "confidence": 98.7},
            {"text": "Property Address: 45, Residency Road, Bangalore - 560025", "x": 80, "y": 250, "width": 420, "height": 25, "confidence": 97.5},
            {"text": "Monthly Income: INR 1,45,000", "x": 80, "y": 300, "width": 240, "height": 20, "confidence": 99.1},
            {"text": "Loan Amount Requested: INR 50,00,000", "x": 80, "y": 330, "width": 300, "height": 20, "confidence": 99.3},
            {"text": "Primary Employer: Tech Mahindra Ltd", "x": 80, "y": 370, "width": 280, "height": 20, "confidence": 98.2}
        ],
        "font_analysis": {
            "status": "Passed",
            "detected_fonts": ["Helvetica", "Arial"],
            "variances": []
        },
        "signature_analysis": {
            "status": "Passed",
            "detected_signatures": 2,
            "confidence": 96.5
        }
    }

    # If the file has a "fraud" or "tampered" name, insert font mismatches and ELA heatmaps
    if "tampered" in basename or "fraud" in basename:
        layout_data["text_blocks"][4] = {"text": "Monthly Income: INR 8,45,000", "x": 80, "y": 300, "width": 240, "height": 20, "confidence": 88.4}
        layout_data["font_analysis"] = {
            "status": "Alert",
            "detected_fonts": ["Helvetica", "Arial", "TimesNewRoman-Patched"],
            "variances": [
                {"text_block": "Monthly Income: INR 8,45,000", "issue": "Anomalous font spacing and family mismatch (Times New Roman overlaid on Helvetica background)"}
            ]
        }
        layout_data["signature_analysis"] = {
            "status": "Alert",
            "detected_signatures": 1,
            "confidence": 62.1,
            "issue": "Co-applicant signature appears digitally copy-pasted (identical transparent pixel grid matched with external documents)."
        }

    # Generate concatenated extracted_text from blocks
    layout_data["extracted_text"] = "\n".join([block["text"] for block in layout_data["text_blocks"]])

    return layout_data
