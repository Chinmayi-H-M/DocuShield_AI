import os
import shutil
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Import our custom modules
from ela import perform_ela, generate_ela_heatmap, detect_suspicious_regions, draw_suspicious_regions
from metadata import extract_image_metadata, extract_pdf_metadata, analyze_metadata_anomalies
from timeline import generate_timeline, generate_human_readable_timeline
from scoring import calculate_authenticity_score, classify_risk, generate_explanation
from report import generate_forensic_report

app = FastAPI(
    title="DocuShield AI — Document Forgery Detection Service",
    description="Microservice API for Error Level Analysis (ELA), PDF structural forensics, and metadata checks.",
    version="1.0.0"
)

# 1. Configure CORS (Cross-Origin Resource Sharing)
# This enables hackathon frontend servers (e.g. Next.js running on port 3000) to query this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
UPLOAD_DIR = "./temp_uploads"
OUTPUT_DIR = "./forensic_output"

# Create directories if they do not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Mount static files directory
# This allows callers to fetch ELA heatmaps and bounding-box images via standard web URLs.
# For example, http://localhost:8000/output/sample_tampered_ela_heatmap.png
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/")
def read_root():
    """Root check endpoint returning system status."""
    return {
        "status": "online",
        "service": "DocuShield AI Document Forensics API Engine",
        "endpoints": {
            "detect": "POST /detect (Ingests PDF/images for forgery analysis)",
            "assets": "GET /output/{filename} (Serves visual report images)"
        }
    }

@app.post("/detect")
async def detect_forgery(file: UploadFile = File(...)):
    """
    Ingests an uploaded document scan (image or PDF), runs ELA, scans metadata,
    reconstructs historical timeline, calculates Authenticity Score, and returns
    a detailed, explainable forensic report in JSON format.
    """
    # 1. Validate file extension
    filename = file.filename
    _, ext = os.path.splitext(filename)
    file_type = ext.lower().replace(".", "")
    
    supported_types = ["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"]
    if file_type not in supported_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_type}'. Supported formats: {', '.join(supported_types)}"
        )
        
    # 2. Save the uploaded file to a temporary location
    temp_file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
    try:
        # 3. Scan Metadata based on file type
        if file_type == "pdf":
            metadata = extract_pdf_metadata(temp_file_path)
        else:
            metadata = extract_image_metadata(temp_file_path)
            
        # 4. Evaluate metadata and structural anomalies
        metadata_analysis = analyze_metadata_anomalies(metadata)
        
        # 5. ELA and Image Region Detection (only if not PDF)
        ela_img = None
        ela_heatmap = None
        bbox_img = None
        suspicious_regions = []
        
        if file_type != "pdf":
            # Run Error Level Analysis
            ela_img = perform_ela(temp_file_path, quality=95, scale_factor=15)
            # Convert to color map heatmap
            ela_heatmap = generate_ela_heatmap(ela_img)
            # Find contours
            suspicious_regions = detect_suspicious_regions(ela_img, threshold_val=20, min_area=100)
            
            # Apply text-ringing heuristic (clear contours if file has zero metadata warnings/anomalies)
            if len(metadata_analysis.get("anomalies", [])) == 0 and len(metadata_analysis.get("warnings", [])) == 0:
                suspicious_regions = []
                
            # Draw contours on original canvas
            bbox_img = draw_suspicious_regions(temp_file_path, suspicious_regions)
            
        # 6. Reconstruct chronological timeline
        timeline_events = generate_timeline(metadata)
        timeline_readable = generate_human_readable_timeline(timeline_events)
        
        # 7. Authenticity scoring and risk rating
        score, deductions = calculate_authenticity_score(len(suspicious_regions), metadata_analysis)
        risk_level = classify_risk(score)
        explanation = generate_explanation(score, risk_level, deductions)
        
        # 8. Save report files and images using report.py
        report_json_path = generate_forensic_report(
            original_file_path=temp_file_path,
            ela_image=ela_img,
            ela_heatmap=ela_heatmap,
            bbox_image=bbox_img,
            metadata_results=metadata,
            metadata_analysis=metadata_analysis,
            score=score,
            risk_level=risk_level,
            explanation=explanation,
            timeline_events=timeline_events,
            timeline_readable=timeline_readable,
            suspicious_regions=len(suspicious_regions),
            deductions=deductions,
            output_dir=OUTPUT_DIR
        )
        
        # 9. Load the written JSON report to return it
        with open(report_json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        # 10. Inject visual output URLs pointing to static mount for frontend convenience
        base_name = os.path.splitext(filename)[0]
        host_url = "http://localhost:8000"  # Default port, can be overridden by frontend
        
        visual_outputs = {
            "timeline_report_url": f"{host_url}/output/{base_name}_timeline_report.txt"
        }
        
        if file_type != "pdf":
            visual_outputs.update({
                "ela_diff_url": f"{host_url}/output/{base_name}_ela_diff.png",
                "ela_heatmap_url": f"{host_url}/output/{base_name}_ela_heatmap.png",
                "suspicious_regions_url": f"{host_url}/output/{base_name}_suspicious_regions.png"
            })
            
        report_data["visual_outputs"] = visual_outputs
        
        return JSONResponse(content=report_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forensics processing failure: {str(e)}")
        
    finally:
        # 11. Clean up temporary uploaded file from directory
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    # Start web server on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
