# app/api/chicken_scans/models.py

from pydantic import BaseModel, Field
from datetime import date

# Ensure the disease labels match the ones used in your frontend chart
DISEASE_LABELS = {
    "avian influenza", "blue comb", "coccidiosis", "coccidiosis poops",
    "fowl cholera", "fowl-pox", "mycotic infections", "salmonela", "healthy"
}

class ScanResultIn(BaseModel):
    """Model for incoming scan data from the frontend."""
    # Use 'Healthy' if no specific disease is detected
    diagnosis: str = Field(..., description="The diagnosis result (must be one of the recognized labels).")
    
    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": "Healthy",
            }
        }

class ScanResultOut(BaseModel):
    """Model for the response data."""
    id: int
    diagnosis: str
    farm_id: int
    timestamp: str # Stored as text/ISO format

class TrendDataResponse(BaseModel):
    """Schema for a single data point in the health trend chart."""
    date: date  # Will be formatted as 'YYYY-MM-DD'
    healthy_count: int
    issue_count: int

    class Config:
        # Allows ORM objects (like SQLAlchemy results) to be converted to Pydantic models
        from_attributes = True