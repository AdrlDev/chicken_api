# app/api/chicken_scans/models.py

from pydantic import BaseModel, Field

# Ensure the disease labels match the ones used in your frontend chart
DISEASE_LABELS = {
    "Avian Influenza", "Blue Comb", "Coccidiosis", "Coccidiosis Poops", 
    "Fowl Cholera", "Fowl-pox", "Mycotic Infections", "Salmo", "Healthy"
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