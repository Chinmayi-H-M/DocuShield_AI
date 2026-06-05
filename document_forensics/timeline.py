from datetime import datetime

def format_timestamp(iso_str: str) -> str:
    """
    Helper function to convert an ISO-8601 timestamp string into a friendly, readable date format.
    
    Example: "2026-06-03T19:08:48+05:30" becomes "2026-06-03 19:08:48 (+05:30)"
    
    Parameters:
        iso_str (str): The ISO formatted timestamp string.
        
    Returns:
        str: A simplified, highly readable timestamp string.
    """
    if not iso_str:
        return "Unknown Date/Time"
        
    try:
        # Check if the string has offset info
        # Split datetime and offset if it exists (e.g. + or - or Z)
        offset = ""
        if "+" in iso_str:
            parts = iso_str.split("+")
            iso_str = parts[0]
            offset = f" (UTC+{parts[1]})"
        elif "-" in iso_str and iso_str.count("-") == 3: # 3 hyphens means date contains negative timezone
            # E.g. 2026-06-03T19:08:48-05:00
            # Let's split on the last hyphen
            parts = iso_str.rsplit("-", 1)
            iso_str = parts[0]
            offset = f" (UTC-{parts[1]})"
        elif iso_str.endswith("Z"):
            iso_str = iso_str.replace("Z", "")
            offset = " (UTC)"
            
        # Parse the remaining part
        dt = datetime.fromisoformat(iso_str[:19])
        return dt.strftime("%Y-%m-%d %H:%M:%S") + offset
    except Exception:
        return iso_str

def generate_timeline(metadata: dict) -> list:
    """
    Constructs a list of document history events sorted chronologically.
    
    Parameters:
        metadata (dict): Standardized metadata from metadata.py.
        
    Returns:
        list: A list of event dictionaries, where each dict has:
              - timestamp: Raw ISO string
              - date_readable: Formatted readable date
              - event_type: E.g., "Creation", "Modification"
              - description: Bullet point summary of what occurred
    """
    events = []
    
    # Extract details
    c_date = metadata.get("creation_date", "")
    m_date = metadata.get("modification_date", "")
    author = metadata.get("author") or "Unknown Author"
    software = metadata.get("software") or "Unknown Software"
    producer = metadata.get("producer") or "Unknown Producer"
    
    # 1. Add Creation Event
    if c_date:
        events.append({
            "timestamp": c_date,
            "date_readable": format_timestamp(c_date),
            "event_type": "Document Created",
            "description": f"File initialized. Creator/Author: '{author}', Software used: '{software}'."
        })
        
    # 2. Add Modification Event (if modification date is different from creation date)
    if m_date:
        # Check if creation and modification dates are different
        # We compare the first 19 characters (ignores timezone differences in simple comparison)
        if not c_date or c_date[:19] != m_date[:19]:
            # Construct modification details
            mod_desc = "File modified/saved."
            if producer and producer != "Unknown Producer":
                mod_desc += f" Producer/Library: '{producer}'."
            if software and software != "Unknown Software" and software != metadata.get("software"):
                mod_desc += f" Saved using: '{software}'."
                
            events.append({
                "timestamp": m_date,
                "date_readable": format_timestamp(m_date),
                "event_type": "Document Modified",
                "description": mod_desc
            })
            
    # Sort events by timestamp
    # If timestamps fail to compare, they keep the insertion order (Creation first, then Modification)
    try:
        events.sort(key=lambda x: x["timestamp"])
    except Exception:
        pass
        
    return events

def generate_human_readable_timeline(timeline: list) -> str:
    """
    Generates a structured, human-readable ASCII timeline report.
    
    Parameters:
        timeline (list): List of events from generate_timeline.
        
    Returns:
        str: A formatted string representing the document's history.
    """
    if not timeline:
        return "No timeline events could be reconstructed (Missing timestamps)."
        
    lines = []
    lines.append("=" * 60)
    lines.append("DOCUMENT HISTORICAL LIFECYCLE TIMELINE")
    lines.append("=" * 60)
    
    for i, event in enumerate(timeline):
        lines.append(f"[{event['date_readable']}] {event['event_type'].upper()}")
        lines.append(f"  └─ {event['description']}")
        
        # Add visual divider between events, but not after the last event
        if i < len(timeline) - 1:
            lines.append("  │")
            lines.append("  ▼")
            
    lines.append("=" * 60)
    return "\n".join(lines)
