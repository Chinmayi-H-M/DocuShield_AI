import os
import datetime
import logging
import hashlib
import json
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app import models, schemas, security
from app.config import settings
from app.services import forensics, ocr, validator, risk_score, model_inference, layoutlmv3_service, neo4j_service, signature_service, ml_pipeline
logger = logging.getLogger("docushield.pipeline")

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=List[schemas.DocumentAnalysisResponse])
def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(security.RoleChecker(["Admin", "Underwriter"])),
    db: Session = Depends(get_db)
):
    results = []
    
    for upload_file in files:
        # Structured log: upload started
        logger.info(json.dumps({
            "event": "upload_started",
            "file_name": upload_file.filename,
            "user": current_user.username
        }))

        # 1. Read file and compute hash
        file_bytes = upload_file.file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Check if hash already analyzed to avoid redundancy
        existing_doc = db.query(models.Document).filter(models.Document.file_hash == file_hash).first()
        if existing_doc:
            # Audit log warning duplicate upload
            db.add(models.AuditLog(
                username=current_user.username,
                event=f"Duplicate file uploaded: {upload_file.filename} (Matched ID: {existing_doc.id})",
                status="Warn"
            ))
            db.commit()
            
            # Reconstruct the DocumentAnalysisResponse structure for the duplicate document
            reasons = []
            if existing_doc.explainable_ai_reasons:
                try:
                    reasons = json.loads(existing_doc.explainable_ai_reasons)
                except Exception:
                    reasons = []
            
            meta_json = {}
            if existing_doc.metadata_json:
                try:
                    meta_json = json.loads(existing_doc.metadata_json)
                except Exception:
                    meta_json = {}
                    
            layoutlm_intel = None
            if existing_doc.layoutlm_intelligence:
                try:
                    layoutlm_intel = json.loads(existing_doc.layoutlm_intelligence)
                except Exception:
                    layoutlm_intel = None

            results.append({
                "document_id": existing_doc.document_id or str(uuid.uuid4()),
                "status": existing_doc.analysis_status or "processed",
                "ocr_text": existing_doc.extracted_text or "",
                "metadata": meta_json,
                "risk_score": existing_doc.risk_score or existing_doc.fraud_score or 0.0,
                "risk_level": existing_doc.risk_level or "Low",
                "issues": reasons,
                "layoutlm_intelligence": layoutlm_intel,
                "signature_similarity": existing_doc.signature_similarity,
                "possible_forgery": existing_doc.possible_forgery,
                "gnn_fraud_probability": existing_doc.gnn_fraud_probability,
                "gnn_risk_level": existing_doc.gnn_risk_level
            })
            continue
            
        # 2. Save file
        ext = os.path.splitext(upload_file.filename)[1]
        unique_name = f"{uuid.uuid4()}{ext}"
        saved_path = os.path.join(settings.UPLOAD_DIR, unique_name)
        
        with open(saved_path, "wb") as buffer:
            buffer.write(file_bytes)
            
        # Generate unique document ID
        doc_uuid = str(uuid.uuid4())
            
        # 3. Process ELA (Error Level Analysis) and get numeric score
        ela_path = forensics.run_error_level_analysis(saved_path)
        ela_score = forensics.calculate_ela_score(ela_path)
        
        # Calculate dynamic ELA tamper regions
        detected_regions = forensics.detect_ela_anomalies(ela_path)
        
        # 4. Process Metadata Analysis
        meta_report = forensics.inspect_metadata(saved_path, original_filename=upload_file.filename)
        
        # Run Compression and Image Quality checks
        compress_report = forensics.analyze_compression(saved_path)
        quality_report = forensics.analyze_image_quality(saved_path)
        
        # Structured log: metadata extracted
        logger.info(json.dumps({
            "event": "metadata_extracted",
            "document_id": doc_uuid,
            "file_name": upload_file.filename,
            "metadata_status": meta_report.get("status")
        }))
        
        # 5. Process OCR text & OCR Font/Signature Analysis
        ocr_failed = False
        try:
            ocr_report = ocr.analyze_ocr_layout(saved_path, original_filename=upload_file.filename)
        except Exception as e:
            ocr_report = {"text_blocks": [], "font_analysis": {"status": "Passed", "variances": []}, "signature_analysis": {"status": "Passed"}}
            ocr_failed = True
            
        # Structured log: OCR completed
        logger.info(json.dumps({
            "event": "ocr_completed",
            "document_id": doc_uuid,
            "file_name": upload_file.filename,
            "ocr_failed": ocr_failed
        }))
        
        # 5b. Run AI/ML Document Forgery Classification (ResNet50 / ML pipeline)
        try:
            ml_result = ml_pipeline.predict_document(file_bytes)
            ml_score = ml_result["raw_score"]
            ml_confidence = ml_result["confidence"]
            # Convert prediction dictionary keys to match expected lowercase format
            ml_prediction = {
                "prediction": ml_result.get("prediction", "genuine").lower(),
                "confidence": ml_result.get("confidence", 50.0) / 100.0,
                "risk_level": ml_result.get("risk_level", "Low")
            }
        except Exception as ml_err:
            logger.error(f"ML Classifier prediction failed: {ml_err}")
            ml_prediction = {"prediction": "genuine", "confidence": 0.5, "error": str(ml_err)}
            ml_score = 0.0
            ml_confidence = 50.0

        # 5c. Run LayoutLMv3 Document Intelligence
        layoutlm_intel = None
        try:
            layoutlm_intel = layoutlmv3_service.extract_document_intelligence(saved_path, ocr_report.get("text_blocks", []))
            logger.info(json.dumps({
                "event": "layoutlm_completed",
                "document_id": doc_uuid,
                "file_name": upload_file.filename
            }))
        except Exception as layoutlm_err:
            logger.error(f"LayoutLMv3 intelligence extraction failed: {layoutlm_err}")

        # 5d. Run Graph Syndicate analysis
        ocr_fields = ocr.extract_fields_from_text(ocr_report.get("extracted_text", ""))
        intel_fields = {
            "applicant_name": (layoutlm_intel.get("applicant_name", {}).get("value") if (layoutlm_intel and layoutlm_intel.get("applicant_name")) else ocr_fields["applicant_name"]),
            "address": (layoutlm_intel.get("address", {}).get("value") if (layoutlm_intel and layoutlm_intel.get("address")) else ocr_fields["address"]),
            "property_id": (layoutlm_intel.get("property_id", {}).get("value") if (layoutlm_intel and layoutlm_intel.get("property_id")) else ocr_fields["property_id"]),
            "income": (layoutlm_intel.get("income", {}).get("value") if (layoutlm_intel and layoutlm_intel.get("income")) else ocr_fields["monthly_income"]),
            "document_type": (layoutlm_intel.get("document_type", {}).get("value") if (layoutlm_intel and layoutlm_intel.get("document_type")) else "UNKNOWN")
        }
        
        # Ensure values exist or fallback
        app_name = intel_fields["applicant_name"] or ocr_fields["applicant_name"]
        addr_val = intel_fields["address"] or ocr_fields["address"]
        
        phone_numbers = neo4j_service.extract_phone_numbers(ocr_report.get("extracted_text", ""))
        graph_risk_penalty, graph_reason = neo4j_service.calculate_graph_risk_for_document(
            doc_uuid, app_name, addr_val, phone_numbers, db
        )

        # 5e. Run Signature Verification Analysis
        sig_report = {"signature_similarity": 1.0, "possible_forgery": False}
        try:
            sig_report = signature_service.verify_document_signature(
                saved_path, app_name, ocr_report.get("text_blocks", [])
            )
        except Exception as sig_err:
            logger.error(f"Signature verification failed: {sig_err}")

        # 5f. Run GNN Syndicate Analysis
        gnn_report = {"gnn_fraud_probability": 0.0, "risk_level": "Low"}
        try:
            from app.services import gnn_service
            gnn_report = gnn_service.predict_graph_risk(
                doc_id=doc_uuid,
                applicant_name=app_name,
                address=addr_val,
                phone_numbers=phone_numbers,
                db=db
            )
        except Exception as gnn_err:
            logger.error(f"GNN prediction failed in pipeline: {gnn_err}")

        # 6. Calculate Forensic Risk Score incorporating quality, ELA, compression and local features
        risk_report = risk_score.calculate_risk_score(
            meta_report=meta_report,
            ocr_report=ocr_report,
            ml_prediction=ml_prediction,
            ocr_failed=ocr_failed,
            graph_risk_penalty=graph_risk_penalty,
            graph_reason=graph_reason,
            possible_forgery=sig_report["possible_forgery"],
            signature_similarity=sig_report["signature_similarity"],
            gnn_fraud_probability=gnn_report["gnn_fraud_probability"],
            gnn_risk_level=gnn_report["risk_level"],
            ela_score=ela_score,
            compress_report=compress_report,
            quality_report=quality_report
        )

        # Combined weighted score: 0.6 * ML + 0.4 * Forensics
        fraud_score = ml_pipeline.combine_risk_score(
            ml_score=ml_score,
            forensic_score=risk_report["risk_score"]
        )
        )
        
        # Structured log: risk score generated
        logger.info(json.dumps({
            "event": "risk_score_generated",
            "document_id": doc_uuid,
            "risk_score": fraud_score,
            "risk_level": risk_report["risk_level"]
        }))
        
        # Use ML model prediction confidence for UI
        confidence = ml_confidence
        
        metadata_status = meta_report.get("status", "Passed")
        font_status = ocr_report.get("font_analysis", {}).get("status", "Passed")
        signature_status = ocr_report.get("signature_analysis", {}).get("status", "Passed")
        compression_status = compress_report.get("status", "Passed")
        
        # 8. Create Document record
        doc_record = models.Document(
            file_name=upload_file.filename,
            file_type=upload_file.filename.split('.')[-1].upper(),
            file_path=saved_path,
            file_hash=file_hash,
            fraud_score=round(float(fraud_score), 1),
            confidence_score=round(float(confidence), 1),
            risk_level=risk_report["risk_level"],
            metadata_status=metadata_status,
            font_status=font_status,
            signature_status=signature_status,
            compression_status=compression_status,
            uploaded_by_id=current_user.id,
            tamper_regions=json.dumps(detected_regions),
            extracted_text=ocr_report.get("extracted_text", "Sample text"),
            metadata_json=json.dumps(meta_report),
            explainable_ai_reasons=json.dumps(risk_report["issues"]),
            layoutlm_intelligence=json.dumps(layoutlm_intel) if layoutlm_intel else None,
            signature_similarity=round(float(sig_report["signature_similarity"]), 4),
            possible_forgery=sig_report["possible_forgery"],
            gnn_fraud_probability=round(float(gnn_report["gnn_fraud_probability"]), 4),
            gnn_risk_level=gnn_report["risk_level"],
            
            # New processing pipeline columns
            document_id=doc_uuid,
            upload_time=datetime.datetime.utcnow(),
            analysis_status="processed",
            risk_score=float(fraud_score)
        )
        
        db.add(doc_record)
        db.add(models.AuditLog(
            username=current_user.username,
            event=f"Uploaded and analyzed document: {upload_file.filename}. Risk Score: {fraud_score}%, Risk Level: {risk_report['risk_level']}.",
            status="Success" if risk_report["risk_level"] == "Low" else "Warn"
        ))
        db.commit()
        db.refresh(doc_record)
        
        # Sync to Neo4j graph database
        try:
            neo4j_service.add_document_nodes_and_relationships(doc_record, intel_fields, phone_numbers)
        except Exception as neo_sync_err:
            logger.warning(f"Neo4j sync failed: {neo_sync_err}")
        
        # Structured log: database saved
        logger.info(json.dumps({
            "event": "database_saved",
            "document_id": doc_uuid,
            "db_record_id": doc_record.id
        }))
        
        results.append({
            "document_id": doc_uuid,
            "status": "processed",
            "ocr_text": doc_record.extracted_text,
            "metadata": meta_report,
            "risk_score": float(fraud_score),
            "risk_level": risk_report["risk_level"],
            "issues": risk_report["issues"],
            "layoutlm_intelligence": layoutlm_intel,
            "signature_similarity": sig_report["signature_similarity"],
            "possible_forgery": sig_report["possible_forgery"],
            "gnn_fraud_probability": gnn_report["gnn_fraud_probability"],
            "gnn_risk_level": gnn_report["risk_level"]
        })
        
    return results

@router.get("/", response_model=List[schemas.DocumentResponse])
def list_documents(
    current_user: models.User = Depends(security.RoleChecker(["Admin", "Underwriter"])),
    db: Session = Depends(get_db)
):
    if current_user.role == "Admin":
        docs = db.query(models.Document).order_by(models.Document.uploaded_at.desc()).all()
    else:
        docs = db.query(models.Document).filter(models.Document.uploaded_by_id == current_user.id).order_by(models.Document.uploaded_at.desc()).all()
    return docs

@router.get("/{doc_id}")
def get_document_details(
    doc_id: int,
    current_user: models.User = Depends(security.RoleChecker(["Admin", "Underwriter"])),
    db: Session = Depends(get_db)
):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if current_user.role == "Underwriter" and doc.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document")
        
    # De-serialize JSON properties
    return {
        "id": doc.id,
        "document_id": doc.document_id,
        "file_name": doc.file_name,
        "file_type": doc.file_type,
        "file_path": doc.file_path,
        "fraud_score": doc.fraud_score,
        "risk_score": doc.risk_score,
        "confidence_score": doc.confidence_score,
        "risk_level": doc.risk_level,
        "metadata_status": doc.metadata_status,
        "font_status": doc.font_status,
        "signature_status": doc.signature_status,
        "compression_status": doc.compression_status,
        "uploaded_at": doc.uploaded_at,
        "upload_time": doc.upload_time,
        "analysis_status": doc.analysis_status,
        
        # Decoded strings
        "tamper_regions": json.loads(doc.tamper_regions or "[]"),
        "explainable_ai_reasons": json.loads(doc.explainable_ai_reasons or "[]"),
        "metadata_json": json.loads(doc.metadata_json or "{}"),
        "extracted_text": doc.extracted_text,
        "layoutlm_intelligence": json.loads(doc.layoutlm_intelligence) if doc.layoutlm_intelligence else None,
        "signature_similarity": doc.signature_similarity,
        "possible_forgery": doc.possible_forgery,
        "gnn_fraud_probability": doc.gnn_fraud_probability,
        "gnn_risk_level": doc.gnn_risk_level
    }

@router.post("/cross-validate", response_model=schemas.CrossValidationResponse)
def cross_validate_documents(
    doc_id_1: int = Form(...),
    doc_id_2: int = Form(...),
    current_user: models.User = Depends(security.RoleChecker(["Admin", "Underwriter"])),
    db: Session = Depends(get_db)
):
    doc1 = db.query(models.Document).filter(models.Document.id == doc_id_1).first()
    doc2 = db.query(models.Document).filter(models.Document.id == doc_id_2).first()
    
    if not doc1 or not doc2:
        raise HTTPException(status_code=404, detail="One or both documents do not exist")

    if current_user.role == "Underwriter":
        if doc1.uploaded_by_id != current_user.id or doc2.uploaded_by_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to one or both documents")

    # Map records to validation lists using LayoutLMv3 intelligence if available, otherwise regex ocr
    fields1_ocr = ocr.extract_fields_from_text(doc1.extracted_text or "")
    fields2_ocr = ocr.extract_fields_from_text(doc2.extracted_text or "")
    
    intel1 = json.loads(doc1.layoutlm_intelligence) if doc1.layoutlm_intelligence else {}
    intel2 = json.loads(doc2.layoutlm_intelligence) if doc2.layoutlm_intelligence else {}
    
    doc1_payload = {
        "name": doc1.file_name,
        "applicant_name": intel1.get("applicant_name", {}).get("value") or fields1_ocr["applicant_name"],
        "property_address": intel1.get("address", {}).get("value") or fields1_ocr["address"],
        "property_id": intel1.get("property_id", {}).get("value") or fields1_ocr["property_id"],
        "monthly_income": intel1.get("income", {}).get("value") or fields1_ocr["monthly_income"]
    }
    
    doc2_payload = {
        "name": doc2.file_name,
        "applicant_name": intel2.get("applicant_name", {}).get("value") or fields2_ocr["applicant_name"],
        "property_address": intel2.get("address", {}).get("value") or fields2_ocr["address"],
        "property_id": intel2.get("property_id", {}).get("value") or fields2_ocr["property_id"],
        "monthly_income": intel2.get("income", {}).get("value") or fields2_ocr["monthly_income"]
    }

    report = validator.perform_cross_validation([doc1_payload, doc2_payload])
    
    cross_val = models.CrossValidation(
        primary_document_id=doc_id_1,
        secondary_document_id=doc_id_2,
        name_match=report["name_match"],
        address_match=report["address_match"],
        property_match=report["property_match"],
        financial_match=report["financial_match"],
        discrepancy_report=json.dumps(report["discrepancies"])
    )
    
    db.add(cross_val)
    db.add(models.AuditLog(
        username=current_user.username,
        event=f"Run Cross-Document Validation between Document {doc_id_1} and Document {doc_id_2}",
        status="Success" if report["name_match"] and report["address_match"] else "Warn"
    ))
    db.commit()
    db.refresh(cross_val)
    
    return cross_val

# Simulated report generation routes
@router.get("/{doc_id}/download-pdf")
def download_pdf_report(
    doc_id: int,
    current_user: models.User = Depends(security.RoleChecker(["Admin", "Underwriter"])),
    db: Session = Depends(get_db)
):
    # To keep this zero-dependency and fast, we send a nice text/pdf mock content response
    # Or a dummy report file to verify PDF downloads works.
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if current_user.role == "Underwriter" and doc.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document")
        
    dummy_pdf_content = f"""%PDF-1.4
%MOCK DOCUSHIELD UNDERWRITING REPORT
Document ID: {doc.id}
File Name: {doc.file_name}
Fraud Risk Score: {doc.fraud_score}%
Confidence Level: {doc.confidence_score}%
Risk Level: {doc.risk_level}
---------------------------------------------
Reserve Bank of India Compliance Assessment
Check 1: Font Spacing Continuity - {doc.font_status}
Check 2: Metadata Alteration Checks - {doc.metadata_status}
Check 3: Signature Copy-Paste Scanners - {doc.signature_status}
---------------------------------------------
DocuShield AI Platform Underwriting Security
"""
    temp_pdf_path = os.path.join(settings.UPLOAD_DIR, f"report_{doc.id}.pdf")
    with open(temp_pdf_path, "w", encoding="utf-8") as f:
        f.write(dummy_pdf_content)

    return FileResponse(
        path=temp_pdf_path,
        media_type="application/pdf",
        filename=f"DocuShield_Report_{doc.file_name}.pdf"
    )

@router.get("/{doc_id}/download-excel")
def download_excel_report(
    doc_id: int,
    current_user: models.User = Depends(security.RoleChecker(["Admin", "Underwriter"])),
    db: Session = Depends(get_db)
):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if current_user.role == "Underwriter" and doc.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document")
        
    dummy_excel_content = (
        f"DocuShield Case Report\n"
        f"Document ID, {doc.id}\n"
        f"File Name, {doc.file_name}\n"
        f"Fraud Score, {doc.fraud_score}%\n"
        f"Confidence, {doc.confidence_score}%\n"
        f"Risk Level, {doc.risk_level}\n"
        f"Metadata Status, {doc.metadata_status}\n"
        f"Font Status, {doc.font_status}\n"
        f"Signature Status, {doc.signature_status}\n"
        f"Compression Status, {doc.compression_status}\n"
        f"Uploaded At, {doc.uploaded_at}\n"
    )
    temp_excel_path = os.path.join(settings.UPLOAD_DIR, f"report_{doc.id}.csv")
    with open(temp_excel_path, "w", encoding="utf-8") as f:
        f.write(dummy_excel_content)

    return FileResponse(
        path=temp_excel_path,
        media_type="text/csv",
        filename=f"DocuShield_Case_{doc.file_name}.csv"
    )
