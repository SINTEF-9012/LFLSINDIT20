from typing import List, Optional
from pydantic import BaseModel

class SINDITProperty(BaseModel):
    propertyName: str
    propertyValue: Optional[str] = None
    propertyUnit: Optional[str] = None     # "bar", "rpm", "kW"...
    propertyDescription: Optional[str] = None

class SINDITAsset(BaseModel):
    id: str # ex: "Pump_P401"
    label: str
    assetType: str # "Pump", "Motor", "Sensor"...
    assetDescription: Optional[str]
    properties: List[SINDITProperty]

class SINDITRelationship(BaseModel):
    sourceId: str # id of the source asset
    targetId: str # id of the target
    relationshipType: str                
    # ONLY among :
    # consistsOf, partOf, connectedTo,
    # dependsOn, derivedFrom, monitors,
    # controls, simulates, uses,
    # communicatesWith, isTypeOf
    relationshipDescription: Optional[str]

class SINDITKnowledgeGraph(BaseModel):
    assets: List[SINDITAsset]
    relationships: List[SINDITRelationship]