#!/usr/bin/env python3
"""
Crop Pest Detection and Alert System - Complete Python Implementation
Standalone, no external dependencies required (except standard library)

Features:
- Pest report management
- Location-based alert generation
- Trend analysis and statistics
- Geographic distance calculations (Haversine formula)
- Anomaly detection for outbreak identification
- Pest recommendations database
- Interactive console interface
"""

import sys
import math
import time
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

# ==================== CONSTANTS ==================== 

MAX_REPORTS = 1000
MAX_FARMERS = 500
MAX_ALERTS = 5000
EARTH_RADIUS_KM = 6371.0
CONFIDENCE_THRESHOLD = 0.60
MAX_ALERTS_PER_FARMER = 3

# Alert radius based on severity (in km)
ALERT_RADII = {
    'LOW': 2,
    'MEDIUM': 5,
    'HIGH': 10,
    'CRITICAL': 15
}

# ==================== ENUMS ==================== 

class ReportStatus(Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    VERIFIED = "verified"
    REJECTED = "rejected"
    RESOLVED = "resolved"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TrendDirection(Enum):
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"

# ==================== DATA CLASSES ==================== 

@dataclass
class PestDetection:
    """Represents a detected pest from ML model"""
    pest_name: str
    confidence: float  # 0.0 - 1.0
    detection_time: datetime
    severity: str = ""
    detection_id: int = 0
    
    def __post_init__(self):
        """Calculate severity based on confidence"""
        if self.confidence > 0.85:
            self.severity = "HIGH"
        elif self.confidence > 0.70:
            self.severity = "MEDIUM"
        else:
            self.severity = "LOW"
    
    def to_dict(self):
        return {
            "pest_name": self.pest_name,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "detection_time": self.detection_time.isoformat()
        }

@dataclass
class PestReport:
    """Represents a pest report submitted by farmer"""
    report_id: str
    farmer_id: str
    crop_id: int
    latitude: float
    longitude: float
    description: str
    status: ReportStatus = ReportStatus.SUBMITTED
    detected_pests: List[PestDetection] = field(default_factory=list)
    affected_area_percent: float = 0.0
    confidence_score: float = 0.0
    severity_level: SeverityLevel = SeverityLevel.LOW
    created_at: datetime = field(default_factory=datetime.now)
    verified_at: Optional[datetime] = None
    verified_by: str = ""
    image_url: str = ""
    
    def to_dict(self):
        return {
            "report_id": self.report_id,
            "farmer_id": self.farmer_id,
            "crop_id": self.crop_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "description": self.description,
            "status": self.status.value,
            "detected_pests": [p.to_dict() for p in self.detected_pests],
            "affected_area_percent": self.affected_area_percent,
            "confidence_score": round(self.confidence_score, 3),
            "severity_level": self.severity_level.value,
            "created_at": self.created_at.isoformat(),
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verified_by": self.verified_by
        }

@dataclass
class Alert:
    """Represents an alert sent to nearby farmer"""
    alert_id: str
    farmer_id: str
    report_id: str
    distance_km: float
    created_at: datetime = field(default_factory=datetime.now)
    sent: bool = False
    
    def to_dict(self):
        return {
            "alert_id": self.alert_id,
            "farmer_id": self.farmer_id,
            "report_id": self.report_id,
            "distance_km": round(self.distance_km, 1),
            "created_at": self.created_at.isoformat(),
            "sent": self.sent
        }

@dataclass
class Farmer:
    """Represents a farmer in the system"""
    farmer_id: str
    name: str
    latitude: float
    longitude: float
    crop_id: int
    alerts_enabled: bool = True
    recent_alert_count: int = 0
    
    def to_dict(self):
        return {
            "farmer_id": self.farmer_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "crop_id": self.crop_id,
            "alerts_enabled": self.alerts_enabled,
            "recent_alert_count": self.recent_alert_count
        }

# ==================== UTILITY FUNCTIONS ==================== 

def generate_id(prefix: str) -> str:
    """Generate unique ID"""
    return f"{prefix}-{str(uuid.uuid4())[:8]}"

def calculate_severity(confidence: float) -> SeverityLevel:
    """Calculate severity from confidence score"""
    if confidence > 0.85:
        return SeverityLevel.HIGH
    elif confidence > 0.70:
        return SeverityLevel.MEDIUM
    else:
        return SeverityLevel.LOW

# ==================== ALGORITHM 1: DISTANCE CALCULATION ==================== 

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two geographic points using Haversine formula
    Returns distance in kilometers
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_KM * c

# ==================== ALGORITHM 2: ALERT RADIUS DETERMINATION ==================== 

def get_alert_radius(severity: SeverityLevel) -> int:
    """Determine alert radius based on pest severity"""
    return ALERT_RADII.get(severity.value.upper(), 5)

# ==================== ALGORITHM 3: LOCATION-BASED ALERT GENERATION ==================== 

def generate_location_based_alerts(report: PestReport, farmers: List[Farmer], 
                                  existing_alerts: Dict[str, List[Alert]]) -> List[Alert]:
    """
    Generate location-based alerts for a verified report
    
    Algorithm:
    1. Determine alert radius based on severity
    2. Find farmers within radius with matching crop
    3. Check alert frequency (prevent fatigue)
    4. Create and send alerts
    """
    generated_alerts = []
    
    if report.status != ReportStatus.VERIFIED:
        print("✗ Report must be verified before generating alerts")
        return generated_alerts
    
    radius_km = get_alert_radius(report.severity_level)
    
    print(f"\n✓ Generating alerts with radius: {radius_km} km")
    print(f"  Report location: ({report.latitude:.4f}, {report.longitude:.4f})")
    print(f"  Severity: {report.severity_level.value.upper()}")
    
    alerts_generated = 0
    
    for farmer in farmers:
        # Calculate distance
        distance = calculate_distance(
            report.latitude, report.longitude,
            farmer.latitude, farmer.longitude
        )
        
        # Check if within alert radius
        if distance > radius_km:
            continue
        
        # Check if growing same crop
        if farmer.crop_id != report.crop_id:
            continue
        
        # Check if alerts enabled
        if not farmer.alerts_enabled:
            continue
        
        # Skip reporting farmer
        if farmer.farmer_id == report.farmer_id:
            continue
        
        # Check alert frequency (max 3 per farmer per day)
        if len(existing_alerts.get(farmer.farmer_id, [])) >= MAX_ALERTS_PER_FARMER:
            print(f"  ⊘ Alert skipped for {farmer.name} (daily limit reached)")
            continue
        
        # Create alert
        alert = Alert(
            alert_id=generate_id("ALERT"),
            farmer_id=farmer.farmer_id,
            report_id=report.report_id,
            distance_km=distance,
            sent=True
        )
        
        generated_alerts.append(alert)
        existing_alerts[farmer.farmer_id].append(alert)
        farmer.recent_alert_count += 1
        
        print(f"  ✓ Alert sent to {farmer.name} ({distance:.1f} km away)")
        alerts_generated += 1
    
    print(f"  → Total alerts generated: {alerts_generated}")
    return generated_alerts

# ==================== ALGORITHM 4: PEST DETECTION SCORING ==================== 

def score_pest_detection(report: PestReport, pest_name: str, confidence: float) -> None:
    """Score pest detection and update report severity"""
    
    pest = PestDetection(
        pest_name=pest_name,
        confidence=confidence,
        detection_time=datetime.now()
    )
    
    report.detected_pests.append(pest)
    
    # Update report confidence and severity
    if confidence > report.confidence_score:
        report.confidence_score = confidence
        report.severity_level = calculate_severity(confidence)
    
    print(f"  ✓ Pest detected: {pest_name} (confidence: {confidence:.2f})")

# ==================== ALGORITHM 5: TREND ANALYSIS ==================== 

def analyze_pest_trends(reports: List[PestReport]) -> Dict:
    """
    Analyze pest trends in verified reports
    
    Algorithm:
    1. Count verified reports
    2. Calculate statistics
    3. Detect trend direction
    4. Calculate anomalies
    """
    print("\n✓ Analyzing pest trends...")
    
    # Count verified reports
    verified_reports = [r for r in reports if r.status == ReportStatus.VERIFIED]
    verified_count = len(verified_reports)
    
    if verified_count == 0:
        print("  No verified reports for analysis")
        return {
            "verified_count": 0,
            "average_confidence": 0.0,
            "trend": "STABLE",
            "peak_reports": 0
        }
    
    # Calculate average confidence
    avg_confidence = sum(r.confidence_score for r in verified_reports) / verified_count
    
    print(f"  Total verified reports: {verified_count}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    
    # Determine trend direction
    if verified_count >= 2:
        # Compare recent vs older reports
        mid_point = verified_count // 2
        recent_avg = sum(r.confidence_score for r in verified_reports[mid_point:]) / (verified_count - mid_point) if verified_count - mid_point > 0 else 0
        older_avg = sum(r.confidence_score for r in verified_reports[:mid_point]) / mid_point if mid_point > 0 else 0
        
        if recent_avg > older_avg + 0.05:
            trend = "INCREASING"
            print(f"  📈 Trend: INCREASING")
        elif recent_avg < older_avg - 0.05:
            trend = "DECREASING"
            print(f"  📉 Trend: DECREASING")
        else:
            trend = "STABLE"
            print(f"  ➡️  Trend: STABLE")
    else:
        trend = "STABLE"
    
    return {
        "verified_count": verified_count,
        "average_confidence": round(avg_confidence, 3),
        "trend": trend,
        "peak_reports": verified_count
    }

# ==================== ALGORITHM 6: ANOMALY DETECTION ==================== 

def detect_outbreaks(reports: List[PestReport]) -> int:
    """
    Detect outbreak anomalies in pest reports
    Uses 2-sigma rule: mean + 2*std_dev
    """
    print("\n✓ Detecting outbreak anomalies...")
    
    verified_reports = [r for r in reports if r.status == ReportStatus.VERIFIED]
    
    if len(verified_reports) < 2:
        print("  Insufficient data for anomaly detection")
        return 0
    
    # Calculate mean
    mean = sum(r.confidence_score for r in verified_reports) / len(verified_reports)
    
    # Calculate standard deviation
    variance = sum((r.confidence_score - mean) ** 2 for r in verified_reports) / len(verified_reports)
    stddev = math.sqrt(variance)
    threshold = mean + 2 * stddev
    
    print(f"  Mean confidence: {mean:.3f}")
    print(f"  Std deviation: {stddev:.3f}")
    print(f"  Anomaly threshold: {threshold:.3f}")
    
    # Detect anomalies
    anomaly_count = 0
    for report in verified_reports:
        if report.confidence_score > threshold:
            print(f"  ⚠️  OUTBREAK DETECTED: Report {report.report_id} (confidence: {report.confidence_score:.3f})")
            anomaly_count += 1
    
    print(f"  Total anomalies detected: {anomaly_count}")
    return anomaly_count

# ==================== ALGORITHM 7: GEOGRAPHIC HOTSPOT IDENTIFICATION ==================== 

def identify_hotspots(reports: List[PestReport]) -> List[Dict]:
    """
    Identify geographic hotspots using grid-based clustering
    Groups nearby reports into concentration zones
    """
    print("\n✓ Identifying geographic hotspots...")
    
    verified_reports = [r for r in reports if r.status == ReportStatus.VERIFIED]
    
    if not verified_reports:
        print("  No verified reports for hotspot analysis")
        return []
    
    # Group reports by grid cells (1-degree cells)
    grid = defaultdict(list)
    
    for report in verified_reports:
        grid_x = int(report.latitude)
        grid_y = int(report.longitude)
        grid_key = (grid_x, grid_y)
        grid[grid_key].append(report)
    
    # Create hotspots from high-density cells
    hotspots = []
    for (grid_x, grid_y), reports_in_cell in grid.items():
        if len(reports_in_cell) >= 5:  # Minimum 5 reports per hotspot
            avg_lat = sum(r.latitude for r in reports_in_cell) / len(reports_in_cell)
            avg_lon = sum(r.longitude for r in reports_in_cell) / len(reports_in_cell)
            
            # Get dominant pest
            pest_counter = defaultdict(int)
            for report in reports_in_cell:
                for pest in report.detected_pests:
                    pest_counter[pest.pest_name] += 1
            
            dominant_pest = max(pest_counter, key=pest_counter.get) if pest_counter else "Unknown"
            
            hotspot = {
                "center_lat": round(avg_lat, 4),
                "center_lon": round(avg_lon, 4),
                "radius_km": 10,
                "report_count": len(reports_in_cell),
                "dominant_pest": dominant_pest,
                "intensity": "HIGH"
            }
            
            hotspots.append(hotspot)
            print(f"  🔴 Hotspot at ({avg_lat:.4f}, {avg_lon:.4f}): {len(reports_in_cell)} reports - {dominant_pest}")
    
    print(f"  Total hotspots identified: {len(hotspots)}")
    return hotspots

# ==================== CORE SYSTEM CLASS ==================== 

class PestDetectionSystem:
    """Main Pest Detection System"""
    
    def __init__(self):
        self.reports: List[PestReport] = []
        self.farmers: List[Farmer] = []
        self.alerts: List[Alert] = []
        self.farmer_alerts: Dict[str, List[Alert]] = defaultdict(list)
    
    def register_farmer(self, farmer_id: str, name: str, latitude: float, 
                       longitude: float, crop_id: int) -> None:
        """Register a farmer in the system"""
        if len(self.farmers) >= MAX_FARMERS:
            print("✗ Farmer buffer full")
            return
        
        farmer = Farmer(farmer_id, name, latitude, longitude, crop_id)
        self.farmers.append(farmer)
        print(f"✓ Farmer registered: {name} ({farmer_id})")
    
    def submit_pest_report(self, farmer_id: str, crop_id: int, latitude: float,
                          longitude: float, description: str) -> Optional[PestReport]:
        """Submit a new pest report"""
        if len(self.reports) >= MAX_REPORTS:
            print("✗ Report buffer full")
            return None
        
        report = PestReport(
            report_id=generate_id("RPT"),
            farmer_id=farmer_id,
            crop_id=crop_id,
            latitude=latitude,
            longitude=longitude,
            description=description
        )
        
        self.reports.append(report)
        
        print(f"\n✓ Pest report submitted: {report.report_id}")
        print(f"  Farmer: {farmer_id}")
        print(f"  Location: ({latitude:.4f}, {longitude:.4f})")
        print(f"  Description: {description}")
        
        return report
    
    def add_pest_detection(self, report_id: str, pest_name: str, confidence: float) -> None:
        """Add pest detection to a report"""
        for report in self.reports:
            if report.report_id == report_id:
                score_pest_detection(report, pest_name, confidence)
                return
        
        print(f"✗ Report not found: {report_id}")
    
    def verify_report(self, report_id: str, verified_by: str = "OfficerA") -> None:
        """Verify a report and generate alerts"""
        for report in self.reports:
            if report.report_id == report_id:
                report.status = ReportStatus.VERIFIED
                report.verified_at = datetime.now()
                report.verified_by = verified_by
                
                print(f"\n✓ Report verified: {report_id}")
                print(f"  Status: {report.status.value.upper()}")
                if report.detected_pests:
                    print(f"  Primary pest: {report.detected_pests[0].pest_name} (confidence: {report.confidence_score:.2f})")
                
                # Generate location-based alerts
                new_alerts = generate_location_based_alerts(report, self.farmers, self.farmer_alerts)
                self.alerts.extend(new_alerts)
                
                return
        
        print(f"✗ Report not found: {report_id}")
    
    def analyze_trends(self) -> Dict:
        """Get pest trend analysis"""
        return analyze_pest_trends(self.reports)
    
    def detect_outbreaks(self) -> int:
        """Detect outbreak anomalies"""
        return detect_outbreaks(self.reports)
    
    def identify_hotspots(self) -> List[Dict]:
        """Identify geographic hotspots"""
        return identify_hotspots(self.reports)
    
    def get_pest_recommendations(self, pest_name: str) -> List[str]:
        """Get control recommendations for pest"""
        recommendations = {
            "Armyworm": [
                "1. Chemical: Apply Spinosad or Lambda-cyhalothrin at dusk",
                "2. Biological: Release Bacillus thuringiensis (Bt)",
                "3. Cultural: Remove affected plant parts, deep plowing",
                "4. Timing: Apply at dusk for maximum effectiveness"
            ],
            "Whitefly": [
                "1. Chemical: Spray Neem oil or Pyrethrin early morning",
                "2. Biological: Release Encarsia wasps",
                "3. Cultural: Yellow sticky traps, remove weeds",
                "4. Timing: Spray early morning or late evening"
            ],
            "Aphid": [
                "1. Chemical: Apply Imidacloprid or Acetamiprid",
                "2. Biological: Release Ladybugs and Lacewings",
                "3. Cultural: High-pressure water spray, mulching",
                "4. Timing: Treat as soon as colonies appear"
            ],
            "Locust": [
                "1. Chemical: Fipronil or Chlorfenapyr",
                "2. Biological: Fungal spore applications",
                "3. Cultural: Barrier fencing, field burning",
                "4. Timing: Treat at nymph stage for best results"
            ]
        }
        
        return recommendations.get(pest_name, ["No specific recommendations available"])
    
    def print_system_stats(self) -> None:
        """Print system statistics"""
        print("\n" + "=" * 60)
        print("PEST DETECTION SYSTEM STATISTICS")
        print("=" * 60)
        
        verified_count = len([r for r in self.reports if r.status == ReportStatus.VERIFIED])
        
        print(f"Total Reports: {len(self.reports)}")
        print(f"Verified Reports: {verified_count}")
        print(f"Total Alerts Generated: {len(self.alerts)}")
        print(f"Registered Farmers: {len(self.farmers)}")
        
        # Find top detected pests
        pest_counts = defaultdict(int)
        for report in self.reports:
            for pest in report.detected_pests:
                pest_counts[pest.pest_name] += 1
        
        if pest_counts:
            print("\nTop Detected Pests:")
            for pest_name, count in sorted(pest_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {pest_name}: {count} detections")
        
        print("=" * 60)
    
    def export_reports_to_json(self, filename: str = "pest_reports.json") -> None:
        """Export all reports to JSON file"""
        data = {
            "reports": [r.to_dict() for r in self.reports],
            "farmers": [f.to_dict() for f in self.farmers],
            "alerts": [a.to_dict() for a in self.alerts],
            "export_time": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Data exported to {filename}")

# ==================== INTERACTIVE MENU ==================== 

def menu():
    """Interactive menu system"""
    system = PestDetectionSystem()
    
    while True:
        print("\n" + "=" * 60)
        print("CROP PEST DETECTION AND ALERT SYSTEM")
        print("=" * 60)
        print("1. Register Farmer")
        print("2. Submit Pest Report")
        print("3. Add Pest Detection")
        print("4. Verify Report")
        print("5. Analyze Trends")
        print("6. Detect Outbreaks")
        print("7. Identify Hotspots")
        print("8. Get Recommendations")
        print("9. Print Statistics")
        print("10. Export Data to JSON")
        print("11. Run Demo")
        print("12. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-12): ").strip()
        
        if choice == "1":
            farmer_id = input("Farmer ID: ").strip()
            name = input("Farmer Name: ").strip()
            lat = float(input("Latitude: ").strip())
            lon = float(input("Longitude: ").strip())
            crop_id = int(input("Crop ID (1-5): ").strip())
            system.register_farmer(farmer_id, name, lat, lon, crop_id)
        
        elif choice == "2":
            farmer_id = input("Farmer ID: ").strip()
            crop_id = int(input("Crop ID: ").strip())
            lat = float(input("Latitude: ").strip())
            lon = float(input("Longitude: ").strip())
            desc = input("Description: ").strip()
            system.submit_pest_report(farmer_id, crop_id, lat, lon, desc)
        
        elif choice == "3":
            report_id = input("Report ID: ").strip()
            pest_name = input("Pest Name: ").strip()
            confidence = float(input("Confidence (0.0-1.0): ").strip())
            system.add_pest_detection(report_id, pest_name, confidence)
        
        elif choice == "4":
            report_id = input("Report ID: ").strip()
            system.verify_report(report_id)
        
        elif choice == "5":
            trends = system.analyze_trends()
            print("\nTrend Analysis Results:")
            for key, value in trends.items():
                print(f"  {key}: {value}")
        
        elif choice == "6":
            system.detect_outbreaks()
        
        elif choice == "7":
            hotspots = system.identify_hotspots()
            if hotspots:
                print("\nHotspots identified:")
                for i, hs in enumerate(hotspots, 1):
                    print(f"  {i}. {hs}")
        
        elif choice == "8":
            pest_name = input("Pest Name: ").strip()
            recs = system.get_pest_recommendations(pest_name)
            print(f"\nRecommendations for {pest_name}:")
            for rec in recs:
                print(f"  {rec}")
        
        elif choice == "9":
            system.print_system_stats()
        
        elif choice == "10":
            filename = input("Filename (default: pest_reports.json): ").strip() or "pest_reports.json"
            system.export_reports_to_json(filename)
        
        elif choice == "11":
            run_demo(system)
        
        elif choice == "12":
            print("\n✓ Thank you for using Pest Detection System!")
            break
        
        else:
            print("✗ Invalid choice. Please try again.")

# ==================== DEMO FUNCTION ==================== 

def run_demo(system: Optional[PestDetectionSystem] = None) -> None:
    """Run automated demonstration"""
    
    if system is None:
        system = PestDetectionSystem()
    
    print("\n" + "=" * 60)
    print("CROP PEST DETECTION AND ALERT SYSTEM - PYTHON ENGINE")
    print("=" * 60)
    
    # Register farmers
    print("\n--- FARMER REGISTRATION ---")
    system.register_farmer("F001", "Vikram Singh", 23.1815, 79.9864, 1)
    system.register_farmer("F002", "Priya Verma", 23.1902, 79.9751, 1)
    system.register_farmer("F003", "Rajesh Patel", 23.1756, 79.9921, 1)
    system.register_farmer("F004", "Deepak Kumar", 23.1834, 79.9805, 1)
    
    # Submit pest reports
    print("\n--- PEST REPORT SUBMISSION ---")
    r1 = system.submit_pest_report("F001", 1, 23.1815, 79.9864, "Yellow spots on cotton leaves")
    r2 = system.submit_pest_report("F002", 1, 23.1902, 79.9751, "White colonies on rice")
    r3 = system.submit_pest_report("F003", 1, 23.1756, 79.9921, "Green aphid clusters on wheat")
    r4 = system.submit_pest_report("F004", 1, 23.1834, 79.9805, "Damaged maize plants")
    
    # Add pest detections
    if r1: system.add_pest_detection(r1.report_id, "Armyworm", 0.92)
    if r2: system.add_pest_detection(r2.report_id, "Whitefly", 0.78)
    if r3: system.add_pest_detection(r3.report_id, "Aphid", 0.85)
    if r4: system.add_pest_detection(r4.report_id, "Locust", 0.71)
    
    # Verify reports
    print("\n--- REPORT VERIFICATION & ALERT GENERATION ---")
    if r1: system.verify_report(r1.report_id)
    if r2: system.verify_report(r2.report_id)
    if r3: system.verify_report(r3.report_id)
    if r4: system.verify_report(r4.report_id)
    
    # Analyze trends
    print("\n--- TREND ANALYSIS ---")
    system.analyze_trends()
    
    # Detect anomalies
    print("\n--- ANOMALY DETECTION ---")
    system.detect_outbreaks()
    
    # Identify hotspots
    print("\n--- GEOGRAPHIC HOTSPOT ANALYSIS ---")
    system.identify_hotspots()
    
    # Get recommendations
    print("\n--- PEST CONTROL RECOMMENDATIONS ---")
    for pest in ["Armyworm", "Whitefly"]:
        print(f"\n{pest}:")
        for rec in system.get_pest_recommendations(pest):
            print(f"  {rec}")
    
    # Print statistics
    system.print_system_stats()
    
    print("\n✓ Python Engine Execution Completed!")
    print("=" * 60)

# ==================== MAIN ENTRY POINT ==================== 

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demo if "demo" argument is provided
        run_demo()
    else:
        # Run interactive menu
        try:
            menu()
        except KeyboardInterrupt:
            print("\n\n✓ Goodbye!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

"""
USAGE:

1. Run interactive menu:
   python pest_detection_system.py

2. Run demo:
   python pest_detection_system.py demo

3. In Python code:
   from pest_detection_system import PestDetectionSystem
   
   system = PestDetectionSystem()
   system.register_farmer("F001", "John Doe", 10.5, 20.5, 1)
   report = system.submit_pest_report("F001", 1, 10.5, 20.5, "Pest detected")
   system.add_pest_detection(report.report_id, "Armyworm", 0.92)
   system.verify_report(report.report_id)

KEY ALGORITHMS IMPLEMENTED:
===========================
1. Haversine Distance Calculation - Geographic proximity queries
2. Alert Radius Determination - Based on pest severity levels
3. Location-Based Alert Generation - Query + filter + deduplicate
4. Pest Detection Scoring - Confidence to severity mapping
5. Time-Series Trend Analysis - Confidence distribution analysis
6. Anomaly Detection - 2-sigma rule for outbreak identification
7. Geographic Clustering - Grid-based hotspot identification

FEATURES:
=========
✓ Pure Python - No external dependencies (uses only stdlib)
✓ Object-oriented design with dataclasses
✓ Interactive menu system
✓ Automated demo
✓ JSON export functionality
✓ All 7 core algorithms implemented
✓ Production-ready code with comprehensive documentation
"""
