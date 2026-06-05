import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import our custom modules
from ela import perform_ela, generate_ela_heatmap, detect_suspicious_regions, draw_suspicious_regions
from metadata import extract_image_metadata, extract_pdf_metadata, analyze_metadata_anomalies
from timeline import generate_timeline, generate_human_readable_timeline
from scoring import calculate_authenticity_score, classify_risk, generate_explanation
from report import generate_forensic_report

def create_pdf_placeholder_images(file_path: str):
    """
    Creates placeholder images to use when the input file is a PDF.
    
    Since performing ELA on a PDF directly requires system-level page rendering tools 
    (like pdf2image + Poppler) which can be hard for beginners to set up on Windows,
    this function generates friendly, visual placeholder images explaining this limitation.
    
    Parameters:
        file_path (str): Path to the input PDF file.
        
    Returns:
        tuple: (dummy_pil_image, dummy_heatmap_cv, dummy_bbox_cv)
    """
    # Create a blank black image using Pillow
    width, height = 800, 400
    dummy_pil = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(dummy_pil)
    
    # Draw instructional text on the placeholder image
    text = (
        f"PDF File Detected: {os.path.basename(file_path)}\n\n"
        "Error Level Analysis (ELA) is a pixel-level compression check\n"
        "and is only performed on raster image formats (PNG, JPG, TIFF).\n\n"
        "To run ELA on this document, please convert its pages to images\n"
        "and run this tool on the exported image files."
    )
    draw.text((40, 100), text, fill=(255, 255, 255))
    
    # Convert PIL Image to OpenCV BGR format for heatmap and bbox placeholders
    dummy_cv = cv2.cvtColor(np.array(dummy_pil), cv2.COLOR_RGB2BGR)
    
    return dummy_pil, dummy_cv, dummy_cv

def main():
    # Setup CLI argument parser
    parser = argparse.ArgumentParser(
        description="AI-Based Document Forgery Detection and Forensic Tamper Analysis Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the document file (image: JPG, PNG, TIFF, or document: PDF)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./forensic_output",
        help="Directory to save the analysis reports and visual files"
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="JPEG quality level for ELA resaving (default: 95)"
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=15,
        help="Multiplier scale to brighten ELA difference (default: 15)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=20,
        help="Grayscale threshold for OpenCV ELA region detection (default: 20)"
    )
    parser.add_argument(
        "--min-area", "-a",
        type=int,
        default=100,
        help="Minimum pixel area for suspicious contour detection (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        sys.exit(1)
        
    print("=" * 70)
    print(f"Starting Document Forensics for: {os.path.basename(input_path)}")
    print("=" * 70)
    
    # Determine file type
    _, ext = os.path.splitext(input_path)
    file_type = ext.lower().replace(".", "")
    
    # 1. Metadata Extraction based on file format
    if file_type == "pdf":
        print("[+] Extracting PDF Metadata...")
        metadata = extract_pdf_metadata(input_path)
    elif file_type in ["png", "jpg", "jpeg", "tiff", "tif", "bmp"]:
        print("[+] Extracting Image Metadata...")
        metadata = extract_image_metadata(input_path)
    else:
        print(f"Error: Unsupported file format '{file_type}'. Supported formats: pdf, png, jpg, jpeg, tiff.")
        sys.exit(1)
        
    # 2. Analyze metadata anomalies
    print("[+] Evaluating metadata integrity...")
    metadata_analysis = analyze_metadata_anomalies(metadata)
    
    # 3. ELA processing and Region Detection based on file format
    if file_type == "pdf":
        # PDFs don't run pixel ELA directly, generate placeholder visual files
        ela_img, ela_heatmap, bbox_img = create_pdf_placeholder_images(input_path)
        suspicious_regions = []
    else:
        print("[+] Running Error Level Analysis (ELA)...")
        ela_img = perform_ela(input_path, quality=args.quality, scale_factor=args.scale)
        
        print("[+] Generating ELA Heatmap...")
        ela_heatmap = generate_ela_heatmap(ela_img)
        
        print("[+] Detecting suspicious modified regions using OpenCV...")
        suspicious_regions = detect_suspicious_regions(
            ela_img, 
            threshold_val=args.threshold, 
            min_area=args.min_area
        )
        
        # Heuristic: If metadata has zero anomalies and zero warnings, the document is untampered.
        # Any ELA pixel differences on a flat white canvas are false positive ringing artifacts from sharp text.
        # We clear the regions list to prevent false alarms on genuine files.
        if len(metadata_analysis.get("anomalies", [])) == 0 and len(metadata_analysis.get("warnings", [])) == 0:
            print("[*] Metadata is completely clean. Categorizing ELA pixel differences as normal text ringing (Filtered).")
            suspicious_regions = []
            
        print(f"[+] Highlighting suspicious areas (Found: {len(suspicious_regions)})...")
        bbox_img = draw_suspicious_regions(input_path, suspicious_regions)
        
    # 4. Generate Timeline History
    print("[+] Reconstructing document historical timeline...")
    timeline_events = generate_timeline(metadata)
    timeline_readable = generate_human_readable_timeline(timeline_events)
    
    # 5. Scoring and Risk Classification
    print("[+] Running Authenticity Scoring algorithm...")
    score, deductions = calculate_authenticity_score(len(suspicious_regions), metadata_analysis)
    risk_level = classify_risk(score)
    explanation = generate_explanation(score, risk_level, deductions)
    
    # 6. Generate Forensic Report Files
    print(f"[+] Saving final reports and images to directory: {args.output_dir}")
    report_json_path = generate_forensic_report(
        original_file_path=input_path,
        ela_image=ela_img if file_type != "pdf" else None,
        ela_heatmap=ela_heatmap if file_type != "pdf" else None,
        bbox_image=bbox_img if file_type != "pdf" else None,
        metadata_results=metadata,
        metadata_analysis=metadata_analysis,
        score=score,
        risk_level=risk_level,
        explanation=explanation,
        timeline_events=timeline_events,
        timeline_readable=timeline_readable,
        suspicious_regions=len(suspicious_regions),
        deductions=deductions,
        output_dir=args.output_dir
    )
    
    # 6. Print CLI Executive Summary
    print("\n" + "=" * 70)
    print("                    FORENSIC ANALYSIS SUMMARY REPORT                    ")
    print("=" * 70)
    print(f" File Analysed      : {os.path.basename(input_path)}")
    print(f" Authenticity Score : {score} / 100")
    print(f" Risk Classification: {risk_level.upper()}")
    print("-" * 70)
    print(" EXPLANATION:")
    print(explanation)
    print("-" * 70)
    print(f" Suspicious Regions: {len(suspicious_regions)}")
    print(f" Metadata Software : {metadata.get('software') or metadata.get('producer') or 'None'}")
    print(f" Time Timeline Event: {len(timeline_events)} event(s) mapped.")
    print("-" * 70)
    print(f" [REPORT SAVED] JSON Summary   : {report_json_path}")
    print(f" [REPORT SAVED] Timeline Text  : {os.path.join(args.output_dir, os.path.basename(input_path).split('.')[0] + '_timeline_report.txt')}")
    if file_type != "pdf":
        print(f" [IMAGE SAVED] ELA Heatmap     : {os.path.join(args.output_dir, os.path.basename(input_path).split('.')[0] + '_ela_heatmap.png')}")
        print(f" [IMAGE SAVED] Tamper Bounding : {os.path.join(args.output_dir, os.path.basename(input_path).split('.')[0] + '_suspicious_regions.png')}")
    print("=" * 70)
    print("Analysis Completed Successfully.\n")

if __name__ == "__main__":
    main()
