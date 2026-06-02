import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="Underwriter")  # Admin, Underwriter, Auditor
    is_active = Column(Boolean, default=True)
    otp_secret = Column(String, nullable=True)
    otp_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="uploader")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # PDF, PNG, JPG, TIFF
    file_path = Column(String, nullable=False)
    file_hash = Column(String, unique=True, index=True, nullable=False)
    
    # Forensic metrics
    fraud_score = Column(Float, default=0.0)      # 0 to 100
    confidence_score = Column(Float, default=0.0)   # 0 to 100
    risk_level = Column(String, default="Low")     # Low, Medium, High, Critical
    
    # Visual canvas overlays & meta checks
    metadata_status = Column(String, default="Passed")  # Passed, Alert, Tampered
    font_status = Column(String, default="Passed")
    signature_status = Column(String, default="Passed")
    compression_status = Column(String, default="Passed")
    
    # Storage and details
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    uploaded_by_id = Column(Integer, ForeignKey("users.id"))
    
    # New processing pipeline columns
    document_id = Column(String, unique=True, index=True, nullable=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=True)
    analysis_status = Column(String, default="pending", nullable=True)
    risk_score = Column(Float, default=0.0, nullable=True)
    
    # Bounding box coords / heatmap overlays (JSON-serialized strings for simple SQL support)
    tamper_regions = Column(Text, nullable=True)  # JSON representation of bounding boxes & risk levels
    extracted_text = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)
    explainable_ai_reasons = Column(Text, nullable=True) # JSON list of strings

    uploader = relationship("User", back_populates="documents")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    username = Column(String, nullable=False)
    event = Column(String, nullable=False)
    status = Column(String, nullable=False)  # Success, Failure, Warn

class CrossValidation(Base):
    __tablename__ = "cross_validations"

    id = Column(Integer, primary_key=True, index=True)
    primary_document_id = Column(Integer, nullable=False)
    secondary_document_id = Column(Integer, nullable=False)
    
    name_match = Column(Boolean, default=True)
    address_match = Column(Boolean, default=True)
    property_match = Column(Boolean, default=True)
    financial_match = Column(Boolean, default=True)
    
    discrepancy_report = Column(Text, nullable=True)  # JSON-serialized mismatch descriptions
    checked_at = Column(DateTime, default=datetime.datetime.utcnow)
