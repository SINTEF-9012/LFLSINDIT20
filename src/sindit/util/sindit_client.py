import requests
import os
from dotenv import load_dotenv
from ..knowledge_graph.graph_model_for_llm import SINDITKnowledgeGraph
from datetime import datetime

load_dotenv()

KG_NS = "http://sindit.sintef.no/2.0#"

SINDIT_API_URL = os.getenv("SINDIT_API_URL", "http://localhost:9017")
SINDIT_USERNAME = os.getenv("SINDIT_USERNAME", "sindit")
SINDIT_PASSWORD = os.getenv("SINDIT_PASSWORD", "sindit")

class SINDITClient():
    def __init__(self):
        self.session = requests.Session()
        self.base_url = SINDIT_API_URL
        self.token = None
        
    def _authenticate(self):
        try:
            res = self.session.post(
                url=f"{self.base_url}/token",
                data={"username": SINDIT_USERNAME, "password": SINDIT_PASSWORD},
                timeout=10,
            )
            res.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"[SINDITClient] Unable to authenticate at {self.base_url}/token: {e}")

        self.token = res.json().get("access_token")
        self.session.headers.update({"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})

    def post(self, path, payload):
        if not self.token:
            self._authenticate()

        url = f"{self.base_url}/{path}"
        res = self.session.post(url=url, json=payload, timeout=30)

        if res.status_code == 401:
            self._authenticate()
            res = self.session.post(url, json=payload, timeout=30)

        res.raise_for_status()
            
        return res.json()
    
    def get(self, path):
        if not self.token:
            self._authenticate()

        url = f"{self.base_url}/{path}"
        res = self.session.get(url=url, timeout=30)

        if res.status_code == 401:
            self._authenticate()
            res = self.session.get(url, timeout=30)

        res.raise_for_status()
            
        return res.json()

    def query_get_api(self, endpoint: str, uri: str = None, uri_key: str = "node_uri") -> dict:
        """
        Generic GET API call for the SINDIT Knowledge Graph.
        
        - If you provide a URI and a key (such as 'node_uri', 'class_uri', or 'type_uri'), the function will fetch data for that specific node or class.
        - If you do not provide a URI, the function will fetch all nodes.

        Examples:
            # Get all nodes
            query_get_api_with_uri('kg/nodes')

            # Get a specific node by its URI
            query_get_api_with_uri('kg/node', uri="http://sindit.sintef.no/2.0#temperature", uri_key="node_uri")

            # Get all nodes of a specific type
            query_get_api_with_uri('kg/node_types', uri="urn:samm:sindit.sintef.no:1.0.0#AbstractAsset", uri_key="type_uri")
        """

        if uri:
            path = f'{endpoint}?{uri_key}={uri}&depth=2'
        else:
            path = f'{endpoint}'
        response = self.get(path)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}
    
    def query_get_connections_info(self, endpoint: str, uri: str) -> dict:
        """
        Fetches connection information for a specific node URI.
        
        Args:
            endpoint (str): The API endpoint to query.
            uri (str): The URI of the node to fetch connections for.
        
        Returns:
            dict: The JSON response containing connection information.
        """

        path = f'{endpoint}?node_uri={uri}'
        
        response = self.get(path)
        if response.status_code == 200:
            response_body = response.json()
        else:
            raise Exception(f'Error: {response.status_code}: {response.text}')
        
        date_str = response.headers['Date']
        timestamp = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S GMT')
        timestamp_str = timestamp.isoformat() + ' UTC'

        return {
            "name": response_body['label'],
            "host": response_body['host'],
            "port": response_body['port'],
            "isConnected": response_body['isConnected'],
            "timestamp": timestamp_str,
        }
    
    def _to_uri(self, asset_id):
        return f"{KG_NS}/{asset_id}"
    
    def clean_graph(self):
        if not self.token: 
            self._authenticate()
        print("Cleaning Graph")
        self.post("kg/clear", {})

    def store_graph(self, kg: SINDITKnowledgeGraph) -> None:

        if not self.token:
            self._authenticate()

        for asset in kg.assets:
            asset_uri = self._to_uri(asset.id)
            property_uris = []

            # 1. Create each property
            for prop in asset.properties:
                prop_uri = self._to_uri(f"{asset.id}_{prop.propertyName}")
                payload = {
                    "uri": prop_uri,
                    "propertyName": prop.propertyName,
                    "propertyDescription": prop.propertyDescription,
                    "propertyValue": prop.propertyValue,
                    "propertyUnit": prop.propertyUnit,
                }
                # Remove None keys
                payload = {k: v for k, v in payload.items() if v is not None}
                prop_resp = self.post("kg/asset_property", payload)  # ← plus de base_url ici
                print(f"  property {prop.propertyName}: {prop_resp}")
                property_uris.append({"uri": prop_uri})

            # 2. Create the asset with its linked properties
            payload = {
                "uri": asset_uri,
                "label": asset.label,
                "assetType": asset.assetType,
                "assetDescription": asset.assetDescription,
                "assetProperties": property_uris,
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            asset_resp = self.post("kg/asset", payload)  # ← plus de base_url ici
            print(f"asset {asset.label}: {asset_resp}")

        # 3. Create relationships between assets
        for rel in kg.relationships:
            payload = {
                "uri": self._to_uri(f"rel_{rel.sourceId}_{rel.relationshipType}_{rel.targetId}"),
                "relationshipType": rel.relationshipType,
                "relationshipDescription": rel.relationshipDescription,
                "relationshipSource": {"uri": self._to_uri(rel.sourceId)},
                "relationshipTarget": {"uri": self._to_uri(rel.targetId)},
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            rel_resp = self.post("kg/relationship", payload)  # ← plus de base_url ici
            print(f"relation {rel.sourceId} -[{rel.relationshipType}]-> {rel.targetId}: {rel_resp}")