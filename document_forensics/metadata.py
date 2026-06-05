import os
import re
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
from PyPDF2 import PdfReader

def parse_pdf_date(date_str: str) -> str:
    """
    Parses a PDF metadata date string into a standard ISO-8601 format (YYYY-MM-DDTHH:MM:SS+Offset).
    
    PDF dates are commonly formatted as: "D:YYYYMMDDHHmmSSOHH'mm'"
    For example: "D:20260603190848+05'30'"
    
    Parameters:
        date_str (str): The raw PDF date string from metadata.
        
    Returns:
        str: ISO-8601 formatted date string, or the original string if parsing fails.
    """
    if not date_str:
        return ""
    
    # 1. Clean the string by removing the "D:" prefix and any apostrophes
    clean_str = date_str.replace("D:", "").replace("'", "")
    
    # 2. Match the date components using a Regular Expression
    # Pattern looks for: Year (4 digits), Month (2 digits), Day (2 digits),
    # Hour (2 digits), Minute (2 digits), and optional Seconds (2 digits).
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})?", clean_str)
    
    if match:
        parts = match.groups()
        year, month, day, hour, minute = parts[:5]
        second = parts[5] if parts[5] else "00"
        
        # 3. Detect timezone offsets (e.g. +0530, -0800, or Z for UTC)
        tz_match = re.search(r"([+-])(\d{2})(\d{2})?|Z$", clean_str)
        tz_str = ""
        if tz_match:
            if tz_match.group(0) == "Z":
                tz_str = "+00:00"
            else:
                sign = tz_match.group(1)
                tz_h = tz_match.group(2)
                tz_m = tz_match.group(3) if tz_match.group(3) else "00"
                tz_str = f"{sign}{tz_h}:{tz_m}"
                
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}{tz_str}"
        
    return date_str

def parse_exif_date(date_str: str) -> str:
    """
    Parses an EXIF image date string into a standard ISO-8601 format.
    
    EXIF dates are typically formatted as: "YYYY:MM:DD HH:MM:SS"
    
    Parameters:
        date_str (str): The raw EXIF date string.
        
    Returns:
        str: ISO-8601 formatted date string, or the original string if parsing fails.
    """
    if not date_str:
        return ""
    
    try:
        # EXIF uses colons for dates, e.g. "2026:06:03 19:08:48"
        # We parse this pattern and convert it to ISO "2026-06-03T19:08:48"
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except ValueError:
        return date_str

def extract_image_metadata(image_path: str) -> dict:
    """
    Extracts metadata from an image file using PIL.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        dict: Standardized metadata dictionary containing creation_date, modification_date,
              software, author, producer, and all raw tags found.
    """
    metadata = {
        "file_name": os.path.basename(image_path),
        "file_size": os.path.getsize(image_path),
        "file_type": os.path.splitext(image_path)[1].lower().replace(".", ""),
        "creation_date": "",
        "modification_date": "",
        "software": "",
        "author": "",
        "producer": "",
        "raw_metadata": {}
    }
    
    # 1. Fetch OS level timestamps as fallbacks
    # Note: getctime is metadata change time on Unix, but creation time on Windows
    # getmtime is the last modification time of the file content
    try:
        c_time = os.path.getctime(image_path)
        metadata["creation_date"] = datetime.fromtimestamp(c_time).isoformat()
        
        m_time = os.path.getmtime(image_path)
        metadata["modification_date"] = datetime.fromtimestamp(m_time).isoformat()
    except Exception:
        pass

    # 2. Extract embedded EXIF metadata tags
    try:
        with Image.open(image_path) as img:
            info = img.getexif()
            if info:
                for tag_id, value in info.items():
                    # Translate numerical tag IDs to human-readable names (e.g. 305 -> 'Software')
                    tag_name = TAGS.get(tag_id, tag_id)
                    
                    # Decoded bytes to string if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode(errors="replace")
                        except Exception:
                            value = str(value)
                            
                    metadata["raw_metadata"][str(tag_name)] = str(value)
                    
                # 3. Standardize metadata from EXIF fields
                if "Software" in metadata["raw_metadata"]:
                    metadata["software"] = metadata["raw_metadata"]["Software"]
                    
                if "Artist" in metadata["raw_metadata"]:
                    metadata["author"] = metadata["raw_metadata"]["Artist"]
                elif "XPAuthor" in metadata["raw_metadata"]:
                    metadata["author"] = metadata["raw_metadata"]["XPAuthor"]
                    
                # Extract capture dates from EXIF tags
                if "DateTimeOriginal" in metadata["raw_metadata"]:
                    metadata["creation_date"] = parse_exif_date(metadata["raw_metadata"]["DateTimeOriginal"])
                elif "DateTimeDigitized" in metadata["raw_metadata"]:
                    metadata["creation_date"] = parse_exif_date(metadata["raw_metadata"]["DateTimeDigitized"])
                elif "DateTime" in metadata["raw_metadata"]:
                    metadata["creation_date"] = parse_exif_date(metadata["raw_metadata"]["DateTime"])
                    
                # Extract modification date from EXIF tags
                if "DateTime" in metadata["raw_metadata"]:
                    metadata["modification_date"] = parse_exif_date(metadata["raw_metadata"]["DateTime"])
                    
    except Exception as e:
        print(f"Image EXIF extraction warning: {e}")
        
    return metadata

def clean_font_name(font_name: str) -> str:
    """
    Removes PDF subset prefixes (e.g., 'ABCDEF+Helvetica' becomes 'Helvetica') and leading slashes.
    """
    if not font_name:
        return ""
    # Strip leading slash if present
    font_name = font_name.replace("/", "")
    # Remove subset prefix matching 6 uppercase letters + plus symbol
    return re.sub(r'^[A-Z]{6}\+', '', font_name)

def extract_pdf_fonts(pdf_path: str) -> list:
    """
    Traverses PDF pages and inspects their `/Resources` -> `/Font` elements to extract unique font names.
    """
    fonts = set()
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if '/Resources' in page and '/Font' in page['/Resources']:
                font_dict = page['/Resources']['/Font']
                for key in font_dict:
                    font_obj = font_dict[key]
                    if '/BaseFont' in font_obj:
                        cleaned = clean_font_name(str(font_obj['/BaseFont']))
                        if cleaned:
                            fonts.add(cleaned)
    except Exception as e:
        print(f"Font extraction warning: {e}")
    return sorted(list(fonts))

def check_pdf_incremental_updates(pdf_path: str) -> int:
    """
    Scans the raw bytes of a PDF file to count instances of the `%%EOF` marker.
    Each additional marker indicates an incremental update (save/append action).
    """
    try:
        if not os.path.exists(pdf_path):
            return 1
        with open(pdf_path, "rb") as f:
            content = f.read()
        return content.count(b"%%EOF")
    except Exception as e:
        print(f"Incremental updates check warning: {e}")
        return 1

def extract_pdf_metadata(pdf_path: str) -> dict:
    """
    Extracts metadata, embedded fonts, and structural save counts from a PDF file.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        dict: Standardized metadata dictionary including creation_date, modification_date,
              software, author, producer, fonts_used, and incremental_updates.
    """
    metadata = {
        "file_name": os.path.basename(pdf_path),
        "file_size": os.path.getsize(pdf_path),
        "file_type": "pdf",
        "creation_date": "",
        "modification_date": "",
        "software": "",
        "author": "",
        "producer": "",
        "fonts_used": [],
        "incremental_updates": 1,
        "raw_metadata": {}
    }
    
    # 1. Fetch OS level timestamps as fallbacks
    try:
        c_time = os.path.getctime(pdf_path)
        metadata["creation_date"] = datetime.fromtimestamp(c_time).isoformat()
        
        m_time = os.path.getmtime(pdf_path)
        metadata["modification_date"] = datetime.fromtimestamp(m_time).isoformat()
    except Exception:
        pass

    # 2. Extract embedded PDF document info and structure
    try:
        # Extract fonts
        metadata["fonts_used"] = extract_pdf_fonts(pdf_path)
        
        # Check incremental saves
        metadata["incremental_updates"] = check_pdf_incremental_updates(pdf_path)
        
        reader = PdfReader(pdf_path)
        doc_info = reader.metadata
        
        if doc_info:
            for key, val in doc_info.items():
                if val:
                    metadata["raw_metadata"][str(key)] = str(val)
            
            # 3. Standardize metadata from PDF fields
            # Creation Date
            if "/CreationDate" in doc_info:
                metadata["creation_date"] = parse_pdf_date(doc_info["/CreationDate"])
                
            # Modification Date
            if "/ModDate" in doc_info:
                metadata["modification_date"] = parse_pdf_date(doc_info["/ModDate"])
                
            # Software/Creator
            if "/Creator" in doc_info:
                metadata["software"] = doc_info["/Creator"]
                
            # Producer
            if "/Producer" in doc_info:
                metadata["producer"] = doc_info["/Producer"]
                
            # Author
            if "/Author" in doc_info:
                metadata["author"] = doc_info["/Author"]
                
    except Exception as e:
        print(f"PDF metadata extraction warning: {e}")
        
    return metadata

def analyze_metadata_anomalies(metadata: dict) -> dict:
    """
    Analyzes standardized metadata for security warnings and editing indicators.
    
    Checks for:
    - Known editing software signatures (Photoshop, Canva, GIMP, etc.)
    - Mismatch between Creation Date and Modification Date
    - Modification Date occurring BEFORE Creation Date (indicates spoofing)
    - Missing standard metadata fields (Creator, Producer, Author)
    - PDF structural anomalies (Incremental saves / Multiple EOF markers)
    - Font Consistency Warnings (suspicious fonts or excessive unique font counts)
    
    Parameters:
        metadata (dict): The standardized metadata dictionary.
        
    Returns:
        dict: A dictionary containing lists of anomalies and warnings, and structural findings.
    """
    anomalies = []
    warnings = []
    
    software = metadata.get("software", "").lower()
    producer = metadata.get("producer", "").lower()
    
    # 1. Check for editing software signatures
    editing_tools = ["photoshop", "canva", "gimp", "illustrator", "inkscape", "pdfescape", "nitro", "foxit", "affinity"]
    detected_tool = None
    
    for tool in editing_tools:
        if tool in software or tool in producer:
            detected_tool = tool
            break
            
    if detected_tool:
        warnings.append(f"Editing software signature detected: '{metadata.get('software') or metadata.get('producer')}'")
        
    # 2. Check for missing metadata fields
    # For PDFs, Author and Producer are standard fields. If missing, it could indicate stripping or automated generation.
    if metadata.get("file_type") == "pdf":
        if not metadata.get("author"):
            anomalies.append("Missing standard PDF metadata field: 'Author'")
        if not metadata.get("producer"):
            anomalies.append("Missing standard PDF metadata field: 'Producer'")
        if not metadata.get("software"):
            anomalies.append("Missing standard PDF metadata field: 'Creator/Software'")
    else:
        # For images, lack of software/camera signature or artist is normal, but worth noting
        if not metadata.get("software") and not metadata.get("raw_metadata"):
            anomalies.append("Image EXIF metadata is completely stripped or empty")

    # 3. Analyze date anomalies
    c_date_str = metadata.get("creation_date")
    m_date_str = metadata.get("modification_date")
    
    if c_date_str and m_date_str:
        try:
            # Parse ISO formats (handling potential timezones by taking first 19 chars: YYYY-MM-DDTHH:MM:SS)
            c_date = datetime.fromisoformat(c_date_str[:19])
            m_date = datetime.fromisoformat(m_date_str[:19])
            
            # Scenario A: Modification date is before creation date (impossible without manual tampering or spoofing)
            if m_date < c_date:
                anomalies.append("Temporal anomaly: Modification date occurs BEFORE Creation date")
            
            # Scenario B: Modification date is different from creation date
            # A delta greater than 60 seconds indicates the file was edited and resaved after creation.
            delta = abs((m_date - c_date).total_seconds())
            if delta > 60:
                anomalies.append(f"Temporal anomaly: Document modified after creation. Time difference: {delta/3600:.2f} hours.")
                warnings.append(f"Document modified after creation. Time difference: {delta/3600:.2f} hours.")
                
        except Exception as e:
            # If datetime parsing fails, we skip complex date comparison and alert about format issues
            anomalies.append(f"Failed to parse dates for temporal validation: {str(e)}")

    # 4. Advanced PDF & Font Consistency checks
    font_consistency_warnings = []
    pdf_fonts = metadata.get("fonts_used", [])
    incremental_updates_count = metadata.get("incremental_updates", 1)
    has_pdf_structural_warnings = False
    
    if metadata.get("file_type") == "pdf":
        # Check for PDF incremental saves
        if incremental_updates_count > 1:
            anomalies.append(f"Structural anomaly: PDF contains {incremental_updates_count} incremental updates (resaved multiple times)")
            has_pdf_structural_warnings = True
            
        # Font Consistency Analysis
        suspicious_fonts = ["canva", "photoshop", "comic", "brush", "hand", "gimp", "paint"]
        for font in pdf_fonts:
            font_lower = font.lower()
            for susp_f in suspicious_fonts:
                if susp_f in font_lower:
                    font_consistency_warnings.append(f"Experimental Font Warning: Suspicious font family '{font}' found")
                    break
        if len(pdf_fonts) > 5:
            font_consistency_warnings.append(f"Experimental Font Warning: Unusually high number of fonts used ({len(pdf_fonts)} unique fonts)")

    return {
        "anomalies": anomalies,
        "warnings": warnings,
        "editing_software_detected": detected_tool is not None,
        "software_signature": metadata.get("software") or metadata.get("producer") or "Unknown",
        "fonts_used": pdf_fonts,
        "font_consistency_warnings": font_consistency_warnings,
        "incremental_updates_count": incremental_updates_count,
        "has_pdf_structural_warnings": has_pdf_structural_warnings
    }
