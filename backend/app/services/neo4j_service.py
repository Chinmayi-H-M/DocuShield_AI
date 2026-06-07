import os
import re
import json
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app import models

logger = logging.getLogger("docushield.neo4j")

# Connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class Neo4jConnector:
    _instance = None

    def __init__(self):
        self.driver = None
        self.connected = False
        self.error_msg = None
        self._try_connect()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _try_connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Test connection
            self.driver.verify_connectivity()
            self.connected = True
            logger.info("Successfully connected to Neo4j database.")
            self._initialize_schema()
        except Exception as e:
            self.error_msg = str(e)
            self.connected = False
            logger.warning(f"Neo4j connection failed: {e}. Graph services will run in local dynamic fallback mode.")

    def _initialize_schema(self):
        if not self.connected:
            return
        try:
            with self.driver.session() as session:
                # Create constraints for fast unique MERGE operations
                session.run("CREATE CONSTRAINT UNIQUE_APPLICANT IF NOT EXISTS FOR (a:Applicant) REQUIRE a.name IS UNIQUE")
                session.run("CREATE CONSTRAINT UNIQUE_PROPERTY IF NOT EXISTS FOR (p:Property) REQUIRE p.address IS UNIQUE")
                session.run("CREATE CONSTRAINT UNIQUE_PHONE IF NOT EXISTS FOR (ph:Phone) REQUIRE ph.number IS UNIQUE")
                session.run("CREATE CONSTRAINT UNIQUE_DOCUMENT IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE")
                logger.info("Neo4j database constraints initialized successfully.")
        except Exception as schema_err:
            logger.warning(f"Failed to initialize Neo4j constraints: {schema_err}")

    def run_query(self, query: str, parameters: dict = None) -> list:
        if not self.connected:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query execution failed: {e}")
            return []

def extract_phone_numbers(text: str) -> list:
    """Helper to extract standard phone numbers using regex."""
    if not text:
        return []
    # Match standard 10 digit numbers, or formatted with dashes/spaces
    phones = re.findall(r"\b\d{10}\b|\b\d{3}[-\s]\d{3}[-\s]\d{4}\b", text)
    # Normalize by removing non-digits
    return [re.sub(r"\D", "", p) for p in phones]

def extract_co_applicant(text: str) -> str:
    """Helper to extract co-applicant name if present in text."""
    if not text:
        return ""
    match = re.search(r"(?:co-applicant|joint applicant|coapplicant|co-owner)\s*:\s*([^\n\r]+)", text.lower())
    if match:
        name = match.group(1).strip()
        # Clean quotes
        name = re.sub(r'[\'":]', '', name).strip().upper()
        if len(name) > 3 and name != "UNKNOWN":
            return name
    return ""

def add_document_nodes_and_relationships(doc_record: models.Document, fields: dict, phone_numbers: list):
    """
    Called on document upload to sync document data to Neo4j if active.
    """
    connector = Neo4jConnector.get_instance()
    if not connector.connected:
        return

    doc_id = doc_record.document_id or str(doc_record.id)
    applicant_name = fields.get("applicant_name", "").strip().upper()
    address = fields.get("address", "").strip()
    property_id = fields.get("property_id", "").strip().upper()
    income = fields.get("income", 0.0)
    doc_type = fields.get("document_type", doc_record.file_type).upper()
    risk_score = doc_record.fraud_score or doc_record.risk_score or 0.0

    if not applicant_name:
        return

    # Base Cypher parameters
    params = {
        "name": applicant_name,
        "doc_id": doc_id,
        "file_name": doc_record.file_name,
        "file_type": doc_type,
        "risk_score": float(risk_score)
    }

    # 1. Merge Applicant and Document, and create SUBMITTED relation
    q_base = """
    MERGE (a:Applicant {name: $name})
    MERGE (d:Document {document_id: $doc_id})
    SET d.file_name = $file_name, d.file_type = $file_type, d.risk_score = $risk_score
    MERGE (a)-[:SUBMITTED]->(d)
    """
    connector.run_query(q_base, params)

    # 2. Add Property collateral link
    if address or property_id:
        prop_label = property_id or address
        q_prop = """
        MATCH (a:Applicant {name: $name})
        MERGE (p:Property {address: $prop_label})
        SET p.property_id = $property_id, p.raw_address = $address
        MERGE (a)-[:OWNS]->(p)
        """
        connector.run_query(q_prop, {
            "name": applicant_name,
            "prop_label": prop_label,
            "property_id": property_id,
            "address": address
        })

    # 3. Add Phone links
    for phone in phone_numbers:
        if len(phone) >= 10:
            q_phone = """
            MATCH (a:Applicant {name: $name})
            MERGE (ph:Phone {number: $phone})
            MERGE (a)-[:USES_PHONE]->(ph)
            """
            connector.run_query(q_phone, {"name": applicant_name, "phone": phone})

    # 4. Add Co-applicant link
    co_applicant = extract_co_applicant(doc_record.extracted_text)
    if co_applicant and co_applicant != applicant_name:
        q_co = """
        MATCH (a:Applicant {name: $name})
        MERGE (co:Applicant {name: $co_name})
        MERGE (a)-[:CO_APPLICANT]->(co)
        MERGE (co)-[:CO_APPLICANT]->(a)
        """
        connector.run_query(q_co, {"name": applicant_name, "co_name": co_applicant})

def calculate_graph_risk_for_document(
    doc_id: str,
    applicant_name: str,
    address: str,
    phone_numbers: list,
    db: Session,
    income: float = 0.0
) -> tuple:
    """
    Checks if the newly uploaded applicant, property, or phone details overlap
    with other applicants' applications in the database, yielding graph risk penalties.
    """
    if not applicant_name:
        return 0, ""

    penalty = 0
    reasons = []

    # Query all active documents, protecting against SQL NULL comparisons
    query = db.query(models.Document)
    if doc_id:
        query = query.filter((models.Document.document_id != doc_id) | (models.Document.document_id == None))
    existing_docs = query.all()

    shared_property = False
    shared_phone = False
    multi_app = False
    income_discrepancy = False
    discrepancy_amount_old = 0.0

    for doc in existing_docs:
        # Deserialize layout intelligence or fall back to regex ocr fields
        intel = {}
        if doc.layoutlm_intelligence:
            try:
                intel = json.loads(doc.layoutlm_intelligence)
            except Exception:
                intel = {}

        exist_name = (intel.get("applicant_name", {}).get("value") or "").strip().upper()
        exist_addr = (intel.get("address", {}).get("value") or "").strip()
        exist_prop_id = (intel.get("property_id", {}).get("value") or "").strip().upper()
        exist_income = float(intel.get("income", {}).get("value") or 0.0)

        if not exist_name:
            # Fallback to OCR parser fields
            from app.services import ocr
            fields = ocr.extract_fields_from_text(doc.extracted_text or "")
            exist_name = fields["applicant_name"]
            exist_addr = fields["address"]
            exist_prop_id = fields["property_id"]
            exist_income = fields["monthly_income"]

        if not exist_name:
            continue

        # Check: Shared property across DIFFERENT applicants
        if exist_name != applicant_name:
            if address and exist_addr and (address.lower() in exist_addr.lower() or exist_addr.lower() in address.lower()):
                shared_property = True
                reasons.append(f"Shares residential/property address collateral ('{address}') with another applicant '{exist_name}'")
            elif exist_prop_id and exist_prop_id == property_id_clean(doc.extracted_text):
                shared_property = True
                reasons.append(f"Shares Property ID '{exist_prop_id}' with applicant '{exist_name}'")

            # Check: Shared phone number
            exist_phones = extract_phone_numbers(doc.extracted_text)
            for p in phone_numbers:
                if p in exist_phones:
                    shared_phone = True
                    reasons.append(f"Shares phone number '{p}' with another applicant '{exist_name}'")

        # Check: Multiple applications under the same name
        if exist_name == applicant_name:
            multi_app = True
            if exist_income > 0 and income > 0 and abs(exist_income - income) > 1.0:
                income_discrepancy = True
                discrepancy_amount_old = exist_income

    if shared_property:
        penalty += 30
    if shared_phone:
        penalty += 25
    if multi_app:
        penalty += 15
    if income_discrepancy:
        penalty += 45
        reasons.append(f"Income discrepancy: Extracted income (INR {income:,.2f}) deviates from previously submitted salary records (INR {discrepancy_amount_old:,.2f}) for applicant '{applicant_name}'")

    reason_str = "; ".join(reasons) if reasons else ""
    if multi_app and not reasons:
        reason_str = f"Applicant '{applicant_name}' has submitted multiple loan applications"

    return min(penalty, 100), reason_str

def property_id_clean(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"property id\s*:\s*([\w\-]+)", text.lower())
    return match.group(1).strip().upper() if match else ""

def get_graph_network(db: Session) -> dict:
    """
    Returns the network graph containing nodes, edges, and syndicate alerts.
    If Neo4j is offline, dynamically populates from the SQLAlchemy SQL Database.
    """
    connector = Neo4jConnector.get_instance()
    
    # 1. If Neo4j is connected, query the actual DB graph
    if connector.connected:
        try:
            nodes_raw = connector.run_query("MATCH (n) RETURN id(n) as internal_id, labels(n)[0] as type, properties(n) as props")
            rels_raw = connector.run_query("MATCH (n)-[r]->(m) RETURN id(n) as source_id, id(m) as target_id, type(r) as relation")
            
            nodes = []
            node_map = {} # Maps internal ID or name to string ID for links
            
            for nr in nodes_raw:
                int_id = nr["internal_id"]
                ntype = nr["type"]
                props = nr["props"]
                
                # Determine display label
                label = ""
                details = ""
                status = "Clean"
                
                if ntype == "Applicant":
                    label = props.get("name", "Applicant")
                    details = f"Role: Underwriting Applicant"
                    # Query if applicant is linked to fraud elements
                    node_id = f"A_{int_id}"
                elif ntype == "Property":
                    label = props.get("address", "Property")
                    details = f"Collateral registered address"
                    node_id = f"P_{int_id}"
                elif ntype == "Phone":
                    label = props.get("number", "Phone")
                    details = f"Contact phone number"
                    node_id = f"PH_{int_id}"
                elif ntype == "Document":
                    label = props.get("file_name", "Document")
                    risk = props.get("risk_score", 0.0)
                    details = f"Type: {props.get('file_type', 'Unknown')} | Risk: {risk}%"
                    status = "Critical" if risk > 65 else "Suspicious" if risk > 35 else "Clean"
                    node_id = f"D_{int_id}"
                else:
                    label = "Entity"
                    node_id = f"E_{int_id}"
                    
                node_map[int_id] = node_id
                
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "type": ntype,
                    "status": status,
                    "details": details
                })
                
            links = []
            for rr in rels_raw:
                src = node_map.get(rr["source_id"])
                tgt = node_map.get(rr["target_id"])
                if src and tgt:
                    links.append({
                        "source": src,
                        "target": tgt,
                        "relation": rr["relation"]
                    })
                    
            # Compute dynamic status alerts and community alerts in Neo4j
            shared_props = connector.run_query(
                "MATCH (a1:Applicant)-[:OWNS]->(p:Property)<-[:OWNS]-(a2:Applicant) WHERE id(a1) < id(a2) RETURN a1.name as a1, a2.name as a2, p.address as prop"
            )
            shared_phones = connector.run_query(
                "MATCH (a1:Applicant)-[:USES_PHONE]->(ph:Phone)<-[:USES_PHONE]-(a2:Applicant) WHERE id(a1) < id(a2) RETURN a1.name as a1, a2.name as a2, ph.number as phone"
            )
            
            alerts = []
            for sp in shared_props:
                alerts.append(f"Syndicate Alert: Applicants '{sp['a1']}' and '{sp['a2']}' share the exact same property collateral ('{sp['prop']}').")
            for sph in shared_phones:
                alerts.append(f"Syndicate Alert: Applicants '{sph['a1']}' and '{sph['a2']}' share the contact phone number '{sph['phone']}'.")
                
            # If no alerts found, present clean compliance message
            if not alerts:
                alerts.append("No active fraud rings or shared collateral syndicates detected by Neo4j graph analysis.")
                
            return {
                "nodes": nodes,
                "links": links,
                "fraud_alerts": alerts
            }
        except Exception as query_err:
            logger.warning(f"Error querying Neo4j network: {query_err}. Switching to dynamic SQLite fallback.")

    # 2. SQLite / SQLite dynamic graph reconstruction on-the-fly
    docs = db.query(models.Document).all()
    
    nodes = []
    links = []
    
    applicants_registry = {} # name -> id
    properties_registry = {} # address -> id
    phones_registry = {}     # number -> id
    documents_registry = {}   # doc_id -> id
    
    # Track node types count
    applicant_counter = 1
    property_counter = 1
    phone_counter = 1
    doc_counter = 1
    
    # Maps matching details
    shared_property_links = {} # property -> list of applicant names
    shared_phone_links = {}     # phone -> list of applicant names
    
    for doc in docs:
        # Load LayoutLMv3 fields or fallback to OCR extraction
        intel = {}
        if doc.layoutlm_intelligence:
            try:
                intel = json.loads(doc.layoutlm_intelligence)
            except Exception:
                intel = {}
                
        applicant_name = (intel.get("applicant_name", {}).get("value") or "").strip().upper()
        address = (intel.get("address", {}).get("value") or "").strip()
        property_id = (intel.get("property_id", {}).get("value") or "").strip().upper()
        doc_type = (intel.get("document_type", {}).get("value") or doc.file_type).upper()
        
        if not applicant_name:
            from app.services import ocr
            fields = ocr.extract_fields_from_text(doc.extracted_text or "")
            applicant_name = fields["applicant_name"]
            address = fields["address"]
            property_id = fields["property_id"]
            
        if not applicant_name:
            continue
            
        # Register applicant node
        if applicant_name not in applicants_registry:
            node_id = f"A{applicant_counter}"
            applicants_registry[applicant_name] = node_id
            nodes.append({
                "id": node_id,
                "label": applicant_name,
                "type": "Applicant",
                "status": "Clean",
                "details": f"Loan underwriting applicant | Total submitted documents: 1"
            })
            applicant_counter += 1
        else:
            # Increment document count in details
            node_id = applicants_registry[applicant_name]
            for n in nodes:
                if n["id"] == node_id:
                    n["details"] = f"Loan underwriting applicant | Multiple applications"
                    
        # Register document node
        doc_node_id = f"D{doc_counter}"
        documents_registry[doc.document_id or str(doc.id)] = doc_node_id
        nodes.append({
            "id": doc_node_id,
            "label": doc.file_name,
            "type": "Loan",
            "status": "Critical" if doc.fraud_score > 65 else "Suspicious" if doc.fraud_score > 35 else "Clean",
            "details": f"Type: {doc_type} | Risk Score: {doc.fraud_score}%"
        })
        doc_counter += 1
        
        # SUBMITTED relation
        links.append({
            "source": applicants_registry[applicant_name],
            "target": doc_node_id,
            "relation": "SUBMITTED"
        })
        
        # Register Property node
        prop_label = property_id or address
        if prop_label:
            if prop_label not in properties_registry:
                prop_node_id = f"P{property_counter}"
                properties_registry[prop_label] = prop_node_id
                nodes.append({
                    "id": prop_node_id,
                    "label": prop_label[:22] + ("..." if len(prop_label) > 22 else ""),
                    "type": "Property",
                    "status": "Clean",
                    "details": f"Collateral Address: {prop_label}"
                })
                property_counter += 1
            
            # OWNS relation
            links.append({
                "source": applicants_registry[applicant_name],
                "target": properties_registry[prop_label],
                "relation": "OWNS"
            })
            
            # Record collateral shared links
            if prop_label not in shared_property_links:
                shared_property_links[prop_label] = set()
            shared_property_links[prop_label].add(applicant_name)
            
        # Parse and register Phone nodes
        phones = extract_phone_numbers(doc.extracted_text)
        for ph in phones:
            if len(ph) >= 10:
                if ph not in phones_registry:
                    ph_node_id = f"PH{phone_counter}"
                    phones_registry[ph] = ph_node_id
                    nodes.append({
                        "id": ph_node_id,
                        "label": f"+91 {ph[-10:]}",
                        "type": "Phone",
                        "status": "Clean",
                        "details": "Contact phone number"
                    })
                    phone_counter += 1
                
                # USES_PHONE relation
                links.append({
                    "source": applicants_registry[applicant_name],
                    "target": phones_registry[ph],
                    "relation": "USES_PHONE"
                })
                
                # Record phone shared links
                if ph not in shared_phone_links:
                    shared_phone_links[ph] = set()
                shared_phone_links[ph].add(applicant_name)

        # Parse and register Co-applicants
        co_applicant = extract_co_applicant(doc.extracted_text)
        if co_applicant and co_applicant != applicant_name:
            if co_applicant not in applicants_registry:
                co_node_id = f"A{applicant_counter}"
                applicants_registry[co_applicant] = co_node_id
                nodes.append({
                    "id": co_node_id,
                    "label": co_applicant,
                    "type": "Co-applicant",
                    "status": "Suspicious",
                    "details": f"Co-applicant linked to '{applicant_name}'"
                })
                applicant_counter += 1
                
            # CO_APPLICANT links (both directions)
            links.append({
                "source": applicants_registry[applicant_name],
                "target": applicants_registry[co_applicant],
                "relation": "CO_APPLICANT"
            })

    # Fraud Alert detection
    alerts = []
    
    # 1. Evaluate shared properties
    for prop, apps in shared_property_links.items():
        if len(apps) > 1:
            apps_list = list(apps)
            alerts.append(f"Multiple independent applicants ({', '.join(apps_list)}) sharing the exact same property collateral ('{prop}').")
            # Upgrade Property node status to Suspicious or Critical
            prop_id = properties_registry[prop]
            for n in nodes:
                if n["id"] == prop_id:
                    n["status"] = "Critical"
                    n["details"] = f"COLLATERAL FRAUD RING: Collateral shared by {len(apps)} independent applicants."
            # Set applicants status to Critical
            for app in apps_list:
                app_node_id = applicants_registry[app]
                for n in nodes:
                    if n["id"] == app_node_id:
                        n["status"] = "Critical"
                        n["details"] = "CRITICAL RISK: Connected to shared collateral fraud ring."

    # 2. Evaluate shared phone numbers
    for ph, apps in shared_phone_links.items():
        if len(apps) > 1:
            apps_list = list(apps)
            alerts.append(f"Multiple independent applicants ({', '.join(apps_list)}) sharing the contact phone number '+91 {ph[-10:]}'.")
            ph_id = phones_registry[ph]
            for n in nodes:
                if n["id"] == ph_id:
                    n["status"] = "Critical"
                    n["details"] = f"FRAUD RING KEYWORD: Shared by {len(apps)} independent applicants."
            for app in apps_list:
                app_node_id = applicants_registry[app]
                for n in nodes:
                    if n["id"] == app_node_id:
                        n["status"] = "Critical"

    if not alerts:
        alerts.append("No active fraud rings or shared collateral syndicates detected by dynamic graph analysis.")

    return {
        "nodes": nodes,
        "links": links,
        "fraud_alerts": alerts
    }
