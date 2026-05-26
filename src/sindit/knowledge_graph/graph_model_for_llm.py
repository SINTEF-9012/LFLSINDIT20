from typing import List, Optional
from pydantic import BaseModel

class SINDITProperty(BaseModel):
    propertyName: str
    propertyValue: Optional[str] = None
    propertyUnit: Optional[str] = None     # "bar", "rpm", "kW"...
    propertyDescription: Optional[str] = None

class SINDITAsset(BaseModel):
    id: str # ex: "Pompe_P401"
    label: str
    assetType: str # "Pump", "Motor", "Sensor"...
    assetDescription: Optional[str]
    properties: List[SINDITProperty] # paramètres de CET asset

class SINDITRelationship(BaseModel):
    sourceId: str # id de l'asset source
    targetId: str # id de l'asset cible
    relationshipType: str                
    # UNIQUEMENT parmi :
    # consistsOf, partOf, connectedTo,
    # dependsOn, derivedFrom, monitors,
    # controls, simulates, uses,
    # communicatesWith, isTypeOf
    relationshipDescription: Optional[str]

class SINDITKnowledgeGraph(BaseModel):
    assets: List[SINDITAsset]
    relationships: List[SINDITRelationship]