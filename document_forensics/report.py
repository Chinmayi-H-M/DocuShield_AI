import os
import json
import cv2
from PIL import Image

def generate_forensic_report(
    original_file_path: str,
    ela_image: Image.Image,
    ela_heatmap,
    bbox_image,
    metadata_results: dict,
    metadata_analysis: dict,
    score: int,
    risk_level: str,
    explanation: str,
    timeline_events: list,
    timeline_readable: str,
    suspicious_regions: int,
    deductions: list,
    output_dir: str
) -> str:
    """
    Compiles all forensic analysis findings, generates visual files, and writes the JSON report.
    
    Visual Files saved:
    - ELA Heatmap (PNG)
    - Image with Bounding Boxes highlighted (PNG)
    - Timeline report (TXT)
    
    Parameters:
        original_file_path (str): Path to the original input file.
        ela_image (PIL.Image.Image): PIL Image of ELA difference.
        ela_heatmap (np.ndarray): OpenCV image (NumPy array) of the ELA heatmap.
        bbox_image (np.ndarray): OpenCV image (NumPy array) of the image with bounding boxes.
        metadata_results (dict): The standardized metadata dict from metadata.py.
        metadata_analysis (dict): The analysis dict from metadata.py.
        score (int): Authenticity Score (0-100).
        risk_level (str): Classification label.
        explanation (str): Natural language explanation.
        timeline_events (list): Reconstructed chronological events.
        timeline_readable (str): String representation of the timeline.
        suspicious_regions (int): Count of suspicious ELA boxes.
        deductions (list): Deductions applied to the score.
        output_dir (str): Directory where all forensic report outputs will be saved.
        
    Returns:
        str: Path to the generated JSON report.
    """
    # 1. Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get base filename without extension to name output files uniquely
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    
    # 2. Save the visual ELA output files (only if not PDF)
    heatmap_path = ""
    bbox_path = ""
    if ela_heatmap is not None:
        # Save ELA diff image
        ela_diff_path = os.path.join(output_dir, f"{base_name}_ela_diff.png")
        ela_image.save(ela_diff_path)
        
        # Save ELA Heatmap image
        heatmap_path = os.path.join(output_dir, f"{base_name}_ela_heatmap.png")
        cv2.imwrite(heatmap_path, ela_heatmap)
        
    if bbox_image is not None:
        # Save Bounding-box image showing suspicious regions
        bbox_path = os.path.join(output_dir, f"{base_name}_suspicious_regions.png")
        cv2.imwrite(bbox_path, bbox_image)
        
    # 3. Save the human-readable timeline report as a text file
    timeline_path = os.path.join(output_dir, f"{base_name}_timeline_report.txt")
    with open(timeline_path, "w", encoding="utf-8") as f:
        f.write(timeline_readable)
        
    # 4. Construct lists of findings for metadata and ELA
    meta_findings_list = []
    for anomaly in metadata_analysis.get("anomalies", []):
        if "structural" not in anomaly.lower(): # structural goes to pdf_structure_findings
            meta_findings_list.append(f"ANOMALY: {anomaly}")
    for warning in metadata_analysis.get("warnings", []):
        meta_findings_list.append(f"WARNING: {warning}")
    if not meta_findings_list:
        meta_findings_list.append("No metadata anomalies or warnings found.")
        
    ela_findings_list = []
    if ela_heatmap is not None:
        ela_findings_list.append(f"Detected ELA difference pixels above threshold." if score < 90 else "No significant ELA pixel differences detected.")
        ela_findings_list.append(f"Heatmap file: {os.path.basename(heatmap_path)}")
        ela_findings_list.append(f"Bounding boxes file: {os.path.basename(bbox_path)}")
    else:
        ela_findings_list.append("ELA is not performed directly on PDF documents. PDF pages must be rendered to images first.")
        
    # 5. Assemble the final JSON report structure
    final_report = {
        "authenticity_score": score,
        "risk_level": risk_level,
        "metadata_findings": meta_findings_list,
        "ela_findings": ela_findings_list,
        "timeline": {
            "events_count": len(timeline_events),
            "events": timeline_events,
            "report_path": timeline_path
        },
        "pdf_structure_findings": {
            "fonts_used": metadata_analysis.get("fonts_used", []),
            "font_consistency_warnings": metadata_analysis.get("font_consistency_warnings", []),
            "incremental_updates_count": metadata_analysis.get("incremental_updates_count", 1),
            "has_structural_tampering_warnings": metadata_analysis.get("has_pdf_structural_warnings", False)
        },
        "suspicious_regions": suspicious_regions,
        "deductions": deductions,
        "explanation": explanation
    }
    
    # 6. Save the JSON report
    report_json_path = os.path.join(output_dir, f"{base_name}_forensic_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
        
    return report_json_path
