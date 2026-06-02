from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: str = "Underwriter"

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    otp_verified: bool

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class OTPVerify(BaseModel):
    username: str
    otp_code: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str
    otp_required: bool

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class DocumentResponse(BaseModel):
    id: int
    file_name: str
    file_type: str
    fraud_score: float
    confidence_score: float
    risk_level: str
    metadata_status: str
    font_status: str
    signature_status: str
    compression_status: str
    uploaded_at: datetime
    uploaded_by_id: int

    class Config:
        from_attributes = True

class DocumentAnalysisResponse(BaseModel):
    document_id: str
    status: str
    ocr_text: str
    metadata: dict
    risk_score: float
    risk_level: str
    issues: List[str]

    class Config:
        from_attributes = True

class AuditLogResponse(BaseModel):
    id: int
    timestamp: datetime
    username: str
    event: str
    status: str

    class Config:
        from_attributes = True

class CrossValidationResponse(BaseModel):
    id: int
    primary_document_id: int
    secondary_document_id: int
    name_match: bool
    address_match: bool
    property_match: bool
    financial_match: bool
    discrepancy_report: Optional[str] = None
    checked_at: datetime

    class Config:
        from_attributes = True
