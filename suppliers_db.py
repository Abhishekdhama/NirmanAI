import random

# --- Curated Supplier Database ---
# Total 65 suppliers covering major construction materials across Indian states.

SUPPLIERS = [
    # Tier 1 - TMT Steel
    {"name": "Tata Steel Ltd - Jamshedpur", "material_type": "TMT Steel", "state": "Jharkhand", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.98, "avg_lead_days": 10, "price_index": 1.15, "capacity_tons_per_month": 500000, "phone": "+91-9876543201", "gst_registered": True},
    {"name": "JSW Steel Ltd - Bellary", "material_type": "TMT Steel", "state": "Karnataka", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.97, "avg_lead_days": 8, "price_index": 1.10, "capacity_tons_per_month": 400000, "phone": "+91-9876543202", "gst_registered": True},
    {"name": "SAIL - Bhilai", "material_type": "TMT Steel", "state": "Madhya Pradesh", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 12, "price_index": 1.08, "capacity_tons_per_month": 450000, "phone": "+91-9876543203", "gst_registered": True},
    {"name": "Jindal Panther - Raigarh", "material_type": "TMT Steel", "state": "Odisha", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.96, "avg_lead_days": 9, "price_index": 1.12, "capacity_tons_per_month": 350000, "phone": "+91-9876543204", "gst_registered": True},
    {"name": "Essar Steel - Hazira", "material_type": "TMT Steel", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.94, "avg_lead_days": 7, "price_index": 1.11, "capacity_tons_per_month": 380000, "phone": "+91-9876543205", "gst_registered": True},
    # Tier 2 & 3 - TMT Steel
    {"name": "Maharashtra Steel Distributors", "material_type": "TMT Steel", "state": "Maharashtra", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.88, "avg_lead_days": 5, "price_index": 1.02, "capacity_tons_per_month": 20000, "phone": "+91-8876543206", "gst_registered": True},
    {"name": "Chennai TMT Traders", "material_type": "TMT Steel", "state": "Tamil Nadu", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.85, "avg_lead_days": 4, "price_index": 1.05, "capacity_tons_per_month": 15000, "phone": "+91-8876543207", "gst_registered": True},
    {"name": "Pune Iron Works", "material_type": "TMT Steel", "state": "Maharashtra", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.72, "avg_lead_days": 2, "price_index": 0.95, "capacity_tons_per_month": 500, "phone": "+91-7876543208", "gst_registered": False},
    {"name": "Jaipur Steel Syndicate", "material_type": "TMT Steel", "state": "Rajasthan", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.89, "avg_lead_days": 5, "price_index": 1.01, "capacity_tons_per_month": 12000, "phone": "+91-8876543209", "gst_registered": True},
    {"name": "Punjab MetaLink", "material_type": "TMT Steel", "state": "Punjab", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.75, "avg_lead_days": 3, "price_index": 0.98, "capacity_tons_per_month": 800, "phone": "+91-7876543210", "gst_registered": True},

    # Tier 1 - OPC Cement
    {"name": "UltraTech Cement - Mumbai", "material_type": "OPC Cement", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.99, "avg_lead_days": 6, "price_index": 1.20, "capacity_tons_per_month": 600000, "phone": "+91-9876543301", "gst_registered": True},
    {"name": "Ambuja Cements - Gujarat", "material_type": "OPC Cement", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.96, "avg_lead_days": 7, "price_index": 1.15, "capacity_tons_per_month": 500000, "phone": "+91-9876543302", "gst_registered": True},
    {"name": "ACC Limited - Chanda", "material_type": "OPC Cement", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.97, "avg_lead_days": 5, "price_index": 1.18, "capacity_tons_per_month": 550000, "phone": "+91-9876543303", "gst_registered": True},
    {"name": "Shree Cement - Beawar", "material_type": "OPC Cement", "state": "Rajasthan", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 8, "price_index": 1.10, "capacity_tons_per_month": 480000, "phone": "+91-9876543304", "gst_registered": True},
    {"name": "Ramco Cements - Ariyalur", "material_type": "OPC Cement", "state": "Tamil Nadu", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.94, "avg_lead_days": 6, "price_index": 1.12, "capacity_tons_per_month": 420000, "phone": "+91-9876543305", "gst_registered": True},
    {"name": "Dalmia Bharat - Kadapa", "material_type": "OPC Cement", "state": "Telangana", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.93, "avg_lead_days": 7, "price_index": 1.08, "capacity_tons_per_month": 400000, "phone": "+91-9876543306", "gst_registered": True},
    # Tier 2 & 3 - OPC Cement
    {"name": "Hyderabad Cement Suppliers", "material_type": "OPC Cement", "state": "Telangana", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.89, "avg_lead_days": 4, "price_index": 1.05, "capacity_tons_per_month": 25000, "phone": "+91-8876543307", "gst_registered": True},
    {"name": "Surat Cement Depo", "material_type": "OPC Cement", "state": "Gujarat", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.78, "avg_lead_days": 2, "price_index": 0.98, "capacity_tons_per_month": 1000, "phone": "+91-7876543308", "gst_registered": True},
    {"name": "Bengal BuildMart", "material_type": "OPC Cement", "state": "West Bengal", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.85, "avg_lead_days": 5, "price_index": 1.03, "capacity_tons_per_month": 18000, "phone": "+91-8876543309", "gst_registered": True},
    {"name": "Kerala Cement Traders", "material_type": "OPC Cement", "state": "Kerala", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.82, "avg_lead_days": 6, "price_index": 1.06, "capacity_tons_per_month": 15000, "phone": "+91-8876543310", "gst_registered": True},

    # River Sand & Coarse Aggregate
    {"name": "Godavari Sand Miners", "material_type": "River Sand", "state": "Telangana", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.82, "avg_lead_days": 4, "price_index": 1.10, "capacity_tons_per_month": 10000, "phone": "+91-8876543401", "gst_registered": True},
    {"name": "Narmada Sand Co", "material_type": "River Sand", "state": "Madhya Pradesh", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.86, "avg_lead_days": 5, "price_index": 1.05, "capacity_tons_per_month": 12000, "phone": "+91-8876543402", "gst_registered": True},
    {"name": "Ganga River Materials", "material_type": "River Sand", "state": "Uttar Pradesh", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.90, "avg_lead_days": 7, "price_index": 1.12, "capacity_tons_per_month": 25000, "phone": "+91-9876543403", "gst_registered": True},
    {"name": "Local Sand Mafia", "material_type": "River Sand", "state": "Bihar", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.65, "avg_lead_days": 2, "price_index": 0.90, "capacity_tons_per_month": 2000, "phone": "+91-7876543404", "gst_registered": False},
    {"name": "Deccan Aggregates", "material_type": "Coarse Aggregate", "state": "Karnataka", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.88, "avg_lead_days": 4, "price_index": 1.02, "capacity_tons_per_month": 15000, "phone": "+91-8876543405", "gst_registered": True},
    {"name": "Haryana Crushers", "material_type": "Coarse Aggregate", "state": "Haryana", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.84, "avg_lead_days": 3, "price_index": 0.99, "capacity_tons_per_month": 18000, "phone": "+91-8876543406", "gst_registered": True},
    {"name": "Rajputana Stone Crushers", "material_type": "Coarse Aggregate", "state": "Rajasthan", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.92, "avg_lead_days": 6, "price_index": 1.08, "capacity_tons_per_month": 30000, "phone": "+91-9876543407", "gst_registered": True},
    {"name": "Odisha Minerals", "material_type": "Coarse Aggregate", "state": "Odisha", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.87, "avg_lead_days": 5, "price_index": 1.01, "capacity_tons_per_month": 16000, "phone": "+91-8876543408", "gst_registered": True},

    # Bricks & Blocks
    {"name": "EcoBricks India", "material_type": "Fly Ash Bricks", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.91, "avg_lead_days": 7, "price_index": 1.15, "capacity_tons_per_month": 5000, "phone": "+91-9876543501", "gst_registered": True},
    {"name": "Kolkata Clay Works", "material_type": "Fly Ash Bricks", "state": "West Bengal", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.85, "avg_lead_days": 5, "price_index": 1.05, "capacity_tons_per_month": 3000, "phone": "+91-8876543502", "gst_registered": True},
    {"name": "UP Ash Bricks", "material_type": "Fly Ash Bricks", "state": "Uttar Pradesh", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.75, "avg_lead_days": 3, "price_index": 0.95, "capacity_tons_per_month": 1000, "phone": "+91-7876543503", "gst_registered": True},
    {"name": "Magicrete Building Solutions", "material_type": "AAC Blocks", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.96, "avg_lead_days": 8, "price_index": 1.18, "capacity_tons_per_month": 20000, "phone": "+91-9876543504", "gst_registered": True},
    {"name": "Biltech Building Elements", "material_type": "AAC Blocks", "state": "Haryana", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.94, "avg_lead_days": 7, "price_index": 1.12, "capacity_tons_per_month": 15000, "phone": "+91-9876543505", "gst_registered": True},
    {"name": "Siporex India", "material_type": "AAC Blocks", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 6, "price_index": 1.15, "capacity_tons_per_month": 18000, "phone": "+91-9876543506", "gst_registered": True},
    {"name": "Chennai AAC Suppliers", "material_type": "AAC Blocks", "state": "Tamil Nadu", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.88, "avg_lead_days": 4, "price_index": 1.04, "capacity_tons_per_month": 5000, "phone": "+91-8876543507", "gst_registered": True},

    # Structural Steel
    {"name": "Larsen & Toubro Steel", "material_type": "Structural Steel", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.98, "avg_lead_days": 14, "price_index": 1.25, "capacity_tons_per_month": 30000, "phone": "+91-9876543601", "gst_registered": True},
    {"name": "Vizag Profiles", "material_type": "Structural Steel", "state": "Telangana", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.89, "avg_lead_days": 10, "price_index": 1.10, "capacity_tons_per_month": 10000, "phone": "+91-8876543602", "gst_registered": True},
    {"name": "Jindal Structural", "material_type": "Structural Steel", "state": "Haryana", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 12, "price_index": 1.18, "capacity_tons_per_month": 25000, "phone": "+91-9876543603", "gst_registered": True},
    {"name": "Patna Steel Girders", "material_type": "Structural Steel", "state": "Bihar", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.76, "avg_lead_days": 5, "price_index": 0.98, "capacity_tons_per_month": 2000, "phone": "+91-7876543604", "gst_registered": True},

    # Electrical Cable
    {"name": "Finolex Cables", "material_type": "Electrical Cable", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.97, "avg_lead_days": 5, "price_index": 1.15, "capacity_tons_per_month": 20000, "phone": "+91-9876543701", "gst_registered": True},
    {"name": "Polycab India", "material_type": "Electrical Cable", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.96, "avg_lead_days": 6, "price_index": 1.10, "capacity_tons_per_month": 22000, "phone": "+91-9876543702", "gst_registered": True},
    {"name": "Havells India", "material_type": "Electrical Cable", "state": "Uttar Pradesh", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 5, "price_index": 1.12, "capacity_tons_per_month": 21000, "phone": "+91-9876543703", "gst_registered": True},
    {"name": "Kerala Electricals", "material_type": "Electrical Cable", "state": "Kerala", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.86, "avg_lead_days": 4, "price_index": 1.05, "capacity_tons_per_month": 5000, "phone": "+91-8876543704", "gst_registered": True},
    {"name": "Bangalore Wire Corp", "material_type": "Electrical Cable", "state": "Karnataka", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.88, "avg_lead_days": 3, "price_index": 1.02, "capacity_tons_per_month": 6000, "phone": "+91-8876543705", "gst_registered": True},

    # HDPE Pipes
    {"name": "Reliance Industries Pipes", "material_type": "HDPE Pipes", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.99, "avg_lead_days": 10, "price_index": 1.22, "capacity_tons_per_month": 45000, "phone": "+91-9876543801", "gst_registered": True},
    {"name": "Supreme Industries", "material_type": "HDPE Pipes", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.94, "avg_lead_days": 7, "price_index": 1.08, "capacity_tons_per_month": 25000, "phone": "+91-9876543802", "gst_registered": True},
    {"name": "Astral Pipes", "material_type": "HDPE Pipes", "state": "Rajasthan", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.93, "avg_lead_days": 8, "price_index": 1.09, "capacity_tons_per_month": 24000, "phone": "+91-9876543803", "gst_registered": True},
    {"name": "Punjab Pipe Traders", "material_type": "HDPE Pipes", "state": "Punjab", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.74, "avg_lead_days": 3, "price_index": 0.95, "capacity_tons_per_month": 1500, "phone": "+91-7876543804", "gst_registered": True},
    {"name": "Madurai Pipe Distributors", "material_type": "HDPE Pipes", "state": "Tamil Nadu", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.83, "avg_lead_days": 5, "price_index": 1.03, "capacity_tons_per_month": 4000, "phone": "+91-8876543805", "gst_registered": True},

    # Vitrified Tiles
    {"name": "Kajaria Ceramics", "material_type": "Vitrified Tiles", "state": "Rajasthan", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.96, "avg_lead_days": 6, "price_index": 1.18, "capacity_tons_per_month": 15000, "phone": "+91-9876543901", "gst_registered": True},
    {"name": "Somany Ceramics", "material_type": "Vitrified Tiles", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.94, "avg_lead_days": 7, "price_index": 1.14, "capacity_tons_per_month": 14000, "phone": "+91-9876543902", "gst_registered": True},
    {"name": "NITCO Tiles", "material_type": "Vitrified Tiles", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.92, "avg_lead_days": 8, "price_index": 1.15, "capacity_tons_per_month": 12000, "phone": "+91-9876543903", "gst_registered": True},
    {"name": "MP Tile Depot", "material_type": "Vitrified Tiles", "state": "Madhya Pradesh", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.85, "avg_lead_days": 5, "price_index": 1.05, "capacity_tons_per_month": 3000, "phone": "+91-8876543904", "gst_registered": True},
    {"name": "Orissa Tile Mart", "material_type": "Vitrified Tiles", "state": "Odisha", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.77, "avg_lead_days": 4, "price_index": 0.96, "capacity_tons_per_month": 1000, "phone": "+91-7876543905", "gst_registered": True},

    # Plywood
    {"name": "Greenply Industries", "material_type": "Plywood", "state": "West Bengal", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 8, "price_index": 1.16, "capacity_tons_per_month": 12000, "phone": "+91-9876544001", "gst_registered": True},
    {"name": "Century Plyboards", "material_type": "Plywood", "state": "West Bengal", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.94, "avg_lead_days": 7, "price_index": 1.12, "capacity_tons_per_month": 13000, "phone": "+91-9876544002", "gst_registered": True},
    {"name": "Kitply Industries", "material_type": "Plywood", "state": "Uttar Pradesh", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.91, "avg_lead_days": 9, "price_index": 1.08, "capacity_tons_per_month": 10000, "phone": "+91-9876544003", "gst_registered": True},
    {"name": "Kerala Wood Works", "material_type": "Plywood", "state": "Kerala", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.84, "avg_lead_days": 6, "price_index": 1.05, "capacity_tons_per_month": 4000, "phone": "+91-8876544004", "gst_registered": True},
    {"name": "Pune Timber Merchant", "material_type": "Plywood", "state": "Maharashtra", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.72, "avg_lead_days": 3, "price_index": 0.94, "capacity_tons_per_month": 1000, "phone": "+91-7876544005", "gst_registered": False},

    # Paint
    {"name": "Asian Paints", "material_type": "Paint", "state": "Maharashtra", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.98, "avg_lead_days": 5, "price_index": 1.20, "capacity_tons_per_month": 30000, "phone": "+91-9876544101", "gst_registered": True},
    {"name": "Berger Paints", "material_type": "Paint", "state": "West Bengal", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.96, "avg_lead_days": 6, "price_index": 1.15, "capacity_tons_per_month": 28000, "phone": "+91-9876544102", "gst_registered": True},
    {"name": "Nerolac Paints", "material_type": "Paint", "state": "Gujarat", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.95, "avg_lead_days": 5, "price_index": 1.12, "capacity_tons_per_month": 25000, "phone": "+91-9876544103", "gst_registered": True},
    {"name": "Dulux Paints (AkzoNobel)", "material_type": "Paint", "state": "Haryana", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.93, "avg_lead_days": 7, "price_index": 1.10, "capacity_tons_per_month": 22000, "phone": "+91-9876544104", "gst_registered": True},
    {"name": "Indigo Paints", "material_type": "Paint", "state": "Rajasthan", "tier": "Tier 1 (Large Manufacturer)", "reliability_score": 0.90, "avg_lead_days": 6, "price_index": 1.05, "capacity_tons_per_month": 15000, "phone": "+91-9876544105", "gst_registered": True},
    {"name": "Bangalore Paint Distributors", "material_type": "Paint", "state": "Karnataka", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.88, "avg_lead_days": 4, "price_index": 1.02, "capacity_tons_per_month": 5000, "phone": "+91-8876544106", "gst_registered": True},
    {"name": "Patna Paint Shop", "material_type": "Paint", "state": "Bihar", "tier": "Tier 3 (Local Supplier)", "reliability_score": 0.79, "avg_lead_days": 2, "price_index": 0.96, "capacity_tons_per_month": 800, "phone": "+91-7876544107", "gst_registered": True},
    
    # Adding a few more to meet minimum 60
    {"name": "Jabalpur Steel Traders", "material_type": "TMT Steel", "state": "Madhya Pradesh", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.84, "avg_lead_days": 6, "price_index": 1.03, "capacity_tons_per_month": 12000, "phone": "+91-8876543211", "gst_registered": True},
    {"name": "Bhubaneswar Cement Corp", "material_type": "OPC Cement", "state": "Odisha", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.86, "avg_lead_days": 5, "price_index": 1.04, "capacity_tons_per_month": 14000, "phone": "+91-8876543311", "gst_registered": True},
    {"name": "Jharkhand Sand Suppliers", "material_type": "River Sand", "state": "Jharkhand", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.81, "avg_lead_days": 4, "price_index": 1.08, "capacity_tons_per_month": 8000, "phone": "+91-8876543409", "gst_registered": True},
    {"name": "MP Stone Mines", "material_type": "Coarse Aggregate", "state": "Madhya Pradesh", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.83, "avg_lead_days": 5, "price_index": 1.00, "capacity_tons_per_month": 12000, "phone": "+91-8876543410", "gst_registered": True},
    {"name": "Telangana Fly Ash Bricks", "material_type": "Fly Ash Bricks", "state": "Telangana", "tier": "Tier 2 (Regional Distributor)", "reliability_score": 0.82, "avg_lead_days": 6, "price_index": 1.02, "capacity_tons_per_month": 4000, "phone": "+91-8876543508", "gst_registered": True},
]


def find_suppliers(material_type, destination_state=None, min_reliability=0.0, tier=None):
    """
    Find suppliers matching criteria, sorted by reliability score (descending).
    """
    matches = []
    for s in SUPPLIERS:
        if s["material_type"] != material_type:
            continue
        if destination_state and s["state"] != destination_state:
            continue
        if s["reliability_score"] < min_reliability:
            continue
        if tier and s["tier"] != tier:
            continue
        matches.append(s)
        
    # Sort by reliability score
    matches.sort(key=lambda x: x["reliability_score"], reverse=True)
    return matches


def find_alternate_suppliers(material_type, exclude_state=None, top_n=3):
    """
    Find alternate suppliers for a given material, optionally excluding a problematic state.
    """
    alternates = []
    for s in SUPPLIERS:
        if s["material_type"] == material_type:
            if exclude_state and s["state"] == exclude_state:
                continue
            alternates.append(s)
            
    alternates.sort(key=lambda x: x["reliability_score"], reverse=True)
    return alternates[:top_n]


def get_supplier_recommendation(material_type, destination_state, urgency='normal'):
    """
    Recommend primary and backup suppliers based on urgency.
    'high' urgency prioritizes local states or low avg_lead_days.
    'normal' prioritizes high reliability and tier 1.
    """
    all_matching = [s for s in SUPPLIERS if s["material_type"] == material_type]
    
    if not all_matching:
        return {"primary": None, "backups": []}
        
    if urgency == 'high':
        # Sort by lead days (ascending), then reliability
        all_matching.sort(key=lambda x: (x["avg_lead_days"], -x["reliability_score"]))
    else:
        # Sort by reliability (descending)
        all_matching.sort(key=lambda x: (-x["reliability_score"], x["avg_lead_days"]))
        
    # Strongly prefer within state if possible for fast delivery
    in_state = [s for s in all_matching if s["state"] == destination_state]
    out_state = [s for s in all_matching if s["state"] != destination_state]
    
    if urgency == 'high' and in_state:
        primary = in_state[0]
        backups = in_state[1:] + out_state
    else:
        primary = all_matching[0]
        backups = all_matching[1:]
        
    return {
        "primary": primary,
        "backups": backups[:2]  # Max 2 backups
    }


def estimate_shipping_cost(origin_state, destination_state, quantity, material_type):
    """
    Estimate shipping cost based on origin, destination, and quantity.
    Returns a dictionary with details.
    """
    base_rate_per_ton_km = 3.5  # INR
    
    # Dummy distance calculation
    if origin_state == destination_state:
        distance_km = random.randint(50, 200)
    else:
        # Cross-state is generally further
        distance_km = random.randint(300, 1500)
        
    # Bulk discount logic
    discount = 1.0
    if quantity > 5000:
        discount = 0.8
    elif quantity > 1000:
        discount = 0.9
        
    # Special multiplier for some materials
    material_multiplier = 1.0
    if material_type in ["OPC Cement", "Fly Ash Bricks"]:
        material_multiplier = 1.2 # Heavier / more fragile to transport
    elif material_type in ["HDPE Pipes"]:
        material_multiplier = 1.5 # Volumetric weight
        
    estimated_cost = distance_km * base_rate_per_ton_km * quantity * discount * material_multiplier
    
    return {
        "origin_state": origin_state,
        "destination_state": destination_state,
        "quantity_tons": quantity,
        "estimated_distance_km": distance_km,
        "total_shipping_cost_inr": round(estimated_cost, 2)
    }

if __name__ == "__main__":
    print("--- NirmanAI Supplier Database Demo ---")
    
    # 1. Find suppliers
    print("\n[1] Finding TMT Steel suppliers with >0.9 reliability:")
    tmt_suppliers = find_suppliers("TMT Steel", min_reliability=0.9)
    for s in tmt_suppliers[:3]:
        print(f"  - {s['name']} ({s['state']}) | Score: {s['reliability_score']}")
        
    # 2. Get recommendations
    print("\n[2] Recommendations for OPC Cement in Maharashtra (Urgency: High):")
    recs = get_supplier_recommendation("OPC Cement", "Maharashtra", urgency="high")
    if recs["primary"]:
        print(f"  Primary: {recs['primary']['name']} (Lead: {recs['primary']['avg_lead_days']} days)")
    for b in recs["backups"]:
        print(f"  Backup: {b['name']} (Lead: {b['avg_lead_days']} days)")
        
    # 3. Find alternate suppliers
    print("\n[3] Alternate Paint suppliers (Excluding Maharashtra):")
    alt_paints = find_alternate_suppliers("Paint", exclude_state="Maharashtra")
    for s in alt_paints:
        print(f"  - {s['name']} ({s['state']})")
        
    # 4. Estimate shipping
    print("\n[4] Shipping Estimate (TMT Steel from Gujarat to Maharashtra, 2000 tons):")
    estimate = estimate_shipping_cost("Gujarat", "Maharashtra", 2000, "TMT Steel")
    print(f"  Distance: {estimate['estimated_distance_km']} km")
    print(f"  Total Cost: INR {estimate['total_shipping_cost_inr']}")
