package com.agriculturetech.pestdetection.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.agriculturetech.pestdetection.model.*;
import com.agriculturetech.pestdetection.repository.*;
import com.agriculturetech.pestdetection.util.*;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Crop Pest Detection and Alert System
 * Java Backend Service for Pest Report Management
 * 
 * Features:
 * - Accept pest reports with image uploads
 * - Store report metadata in database
 * - Trigger ML pest detection pipeline
 * - Officer verification workflow
 * - Location-based alert generation
 */

@Service
public class PestReportService {

    @Autowired
    private PestReportRepository pestReportRepository;

    @Autowired
    private ImageProcessingService imageProcessingService;

    @Autowired
    private AlertService alertService;

    @Autowired
    private MLPestDetectionClient mlDetectionClient;

    @Autowired
    private GeolocationService geolocationService;

    /**
     * Submit a new pest report with image
     * Input: crop type, location, description, image file
     * Output: Report ID, confidence score, initial recommendations
     */
    public PestReport submitPestReport(
            String farmerId,
            Integer cropId,
            Double latitude,
            Double longitude,
            String description,
            MultipartFile imageFile,
            Float affectedAreaPercent) {

        try {
            // 1. Validate input
            validatePestReportInput(latitude, longitude, imageFile);

            // 2. Upload image to cloud storage
            String imageUrl = imageProcessingService.uploadImage(imageFile, farmerId);

            // 3. Create pest report record
            PestReport report = new PestReport();
            report.setReportId(UUID.randomUUID().toString());
            report.setFarmerId(farmerId);
            report.setCropId(cropId);
            report.setLatitude(latitude);
            report.setLongitude(longitude);
            report.setDescription(description);
            report.setImageUrl(imageUrl);
            report.setAffectedAreaPercent(affectedAreaPercent);
            report.setStatus(ReportStatus.SUBMITTED);
            report.setCreatedAt(LocalDateTime.now());

            // 4. Trigger ML pest detection asynchronously
            triggerPestDetection(report, imageFile);

            // 5. Save report to database
            PestReport savedReport = pestReportRepository.save(report);

            System.out.println("✓ Pest report submitted: " + savedReport.getReportId());
            return savedReport;

        } catch (Exception e) {
            System.err.println("✗ Error submitting pest report: " + e.getMessage());
            throw new RuntimeException("Failed to submit pest report", e);
        }
    }

    /**
     * Trigger ML-based pest detection
     * Sends image to ML service for analysis
     */
    private void triggerPestDetection(PestReport report, MultipartFile imageFile) {
        Thread.ofVirtual().start(() -> {
            try {
                // Call ML service for inference
                DetectionResult detectionResult = mlDetectionClient.detectPests(
                    imageFile,
                    report.getCropId()
                );

                // Update report with detection results
                report.setDetectedPests(detectionResult.getDetectedPests());
                report.setConfidenceScore(detectionResult.getPrimaryConfidence());
                report.setSeverityLevel(calculateSeverity(detectionResult.getPrimaryConfidence()));
                report.setProcessedAt(LocalDateTime.now());

                // Save updated report
                pestReportRepository.save(report);

                System.out.println("✓ ML detection completed for report: " + report.getReportId()
                    + " | Primary pest: " + detectionResult.getPrimaryPest()
                    + " | Confidence: " + String.format("%.2f", detectionResult.getPrimaryConfidence()));

            } catch (Exception e) {
                System.err.println("✗ ML detection failed: " + e.getMessage());
            }
        });
    }

    /**
     * Officer verification of pest report
     * After verification, alerts are generated to nearby farmers
     */
    public void verifyPestReport(String reportId, Boolean isVerified, String officerId, String notes) {
        try {
            PestReport report = pestReportRepository.findById(reportId)
                .orElseThrow(() -> new RuntimeException("Report not found"));

            if (isVerified) {
                report.setStatus(ReportStatus.VERIFIED);
                report.setVerifiedBy(officerId);
                report.setVerificationNotes(notes);
                report.setVerifiedAt(LocalDateTime.now());

                // Generate location-based alerts
                generateLocationBasedAlerts(report);

                System.out.println("✓ Report verified and alerts generated: " + reportId);
            } else {
                report.setStatus(ReportStatus.REJECTED);
                report.setVerificationNotes(notes);
                System.out.println("✓ Report rejected: " + reportId);
            }

            pestReportRepository.save(report);

        } catch (Exception e) {
            System.err.println("✗ Error verifying report: " + e.getMessage());
        }
    }

    /**
     * Generate location-based alerts for nearby farmers
     * Algorithm:
     * 1. Determine alert radius based on severity
     * 2. Query farmers within radius
     * 3. Filter by crop type and preferences
     * 4. Send alerts (deduplicate to prevent fatigue)
     */
    private void generateLocationBasedAlerts(PestReport report) {
        try {
            // Determine alert radius based on severity
            int radiusKm = determineAlertRadius(report.getSeverityLevel());

            // Query nearby farmers
            List<Farmer> nearbyFarmers = geolocationService.findNearbyFarmers(
                report.getLatitude(),
                report.getLongitude(),
                radiusKm,
                report.getCropId()
            );

            System.out.println("  → Found " + nearbyFarmers.size() + " farmers within " + radiusKm + "km radius");

            // Generate alerts (with deduplication)
            int alertsGenerated = 0;
            for (Farmer farmer : nearbyFarmers) {
                // Check recent alert frequency (max 3 per day)
                if (!alertService.hasRecentAlerts(farmer.getFarmerId(), 3)) {
                    Alert alert = new Alert();
                    alert.setAlertId(UUID.randomUUID().toString());
                    alert.setFarmerId(farmer.getFarmerId());
                    alert.setReportId(report.getReportId());
                    alert.setDistanceKm(calculateDistance(
                        report.getLatitude(), report.getLongitude(),
                        farmer.getLatitude(), farmer.getLongitude()
                    ));
                    alert.setCreatedAt(LocalDateTime.now());
                    alert.setSent(false);

                    // Send push notification asynchronously
                    alertService.sendPushNotification(farmer, report, alert.getDistanceKm());
                    alertsGenerated++;
                }
            }

            System.out.println("  → " + alertsGenerated + " alerts generated (deduplicated)");

        } catch (Exception e) {
            System.err.println("✗ Error generating alerts: " + e.getMessage());
        }
    }

    /**
     * Get pest trends for a region and crop
     * Returns aggregated data: time-series, hotspots, seasonal patterns
     */
    public PestTrendAnalysis getPestTrends(String region, Integer cropId, Integer daysBack) {
        try {
            LocalDateTime startDate = LocalDateTime.now().minusDays(daysBack);

            // Query verified reports for the period
            List<PestReport> historicalReports = pestReportRepository.findByRegionAndCropAndDateRange(
                region, cropId, startDate, LocalDateTime.now()
            );

            PestTrendAnalysis analysis = new PestTrendAnalysis();

            // 1. Time-series aggregation
            Map<String, Integer> dailyTrends = aggregateDailyTrends(historicalReports);
            analysis.setDailyTrends(dailyTrends);

            // 2. Detect anomalies (outbreak threshold)
            List<OutbreakAlert> outbreaks = detectOutbreaks(dailyTrends);
            analysis.setDetectedOutbreaks(outbreaks);

            // 3. Identify pest hotspots using clustering
            List<Hotspot> hotspots = identifyHotspots(historicalReports);
            analysis.setHotspots(hotspots);

            // 4. Calculate trend direction
            TrendDirection direction = calculateTrendDirection(dailyTrends);
            analysis.setTrendDirection(direction);

            System.out.println("✓ Pest trends analyzed for region: " + region + " crop: " + cropId);
            return analysis;

        } catch (Exception e) {
            System.err.println("✗ Error analyzing trends: " + e.getMessage());
            throw new RuntimeException("Trend analysis failed", e);
        }
    }

    /**
     * Get pest control recommendations for detected pests
     */
    public List<ControlMeasure> getPestControlRecommendations(String pestName, Integer cropId) {
        try {
            // Query control measures from database
            List<ControlMeasure> measures = pestReportRepository.findControlMeasures(pestName, cropId);

            // Rank by effectiveness
            measures.sort((a, b) -> Float.compare(b.getEffectiveness(), a.getEffectiveness()));

            System.out.println("✓ Fetched " + measures.size() + " control recommendations for " + pestName);
            return measures;

        } catch (Exception e) {
            System.err.println("✗ Error fetching recommendations: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    /**
     * Search pest reports by location and date range
     */
    public List<PestReport> searchPestReports(Double latitude, Double longitude, 
            Integer radiusKm, LocalDateTime startDate, LocalDateTime endDate) {
        try {
            List<PestReport> results = pestReportRepository.findByLocationAndDateRange(
                latitude, longitude, radiusKm, startDate, endDate
            );

            System.out.println("✓ Found " + results.size() + " pest reports in search area");
            return results;

        } catch (Exception e) {
            System.err.println("✗ Search failed: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    /**
     * Track pesticide usage by farmer
     */
    public void logPesticideUsage(String farmerId, String pesticideName, 
            Float quantity, String unit, String notes) {
        try {
            PesticideLog log = new PesticideLog();
            log.setLogId(UUID.randomUUID().toString());
            log.setFarmerId(farmerId);
            log.setPesticideName(pesticideName);
            log.setQuantity(quantity);
            log.setUnit(unit);
            log.setNotes(notes);
            log.setApplicationDate(LocalDateTime.now());

            // Save to database
            pestReportRepository.savePesticideLog(log);

            System.out.println("✓ Pesticide usage logged: " + quantity + unit + " of " + pesticideName);

        } catch (Exception e) {
            System.err.println("✗ Error logging pesticide usage: " + e.getMessage());
        }
    }

    // ==================== HELPER METHODS ====================

    private void validatePestReportInput(Double latitude, Double longitude, MultipartFile imageFile) {
        if (latitude == null || longitude == null || imageFile == null || imageFile.isEmpty()) {
            throw new IllegalArgumentException("Missing required fields: location and image");
        }
        if (imageFile.getSize() > 10 * 1024 * 1024) { // 10MB limit
            throw new IllegalArgumentException("Image size exceeds 10MB limit");
        }
    }

    private SeverityLevel calculateSeverity(Float confidence) {
        if (confidence > 0.85) return SeverityLevel.HIGH;
        else if (confidence > 0.70) return SeverityLevel.MEDIUM;
        else return SeverityLevel.LOW;
    }

    private int determineAlertRadius(SeverityLevel severity) {
        switch (severity) {
            case CRITICAL: return 15;
            case HIGH: return 10;
            case MEDIUM: return 5;
            default: return 2;
        }
    }

    private double calculateDistance(Double lat1, Double lon1, Double lat2, Double lon2) {
        // Haversine formula for distance calculation
        double R = 6371; // Earth radius in km
        double dLat = Math.toRadians(lat2 - lat1);
        double dLon = Math.toRadians(lon2 - lon1);
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                   Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                   Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    private Map<String, Integer> aggregateDailyTrends(List<PestReport> reports) {
        return reports.stream()
            .collect(Collectors.groupingBy(
                r -> r.getCreatedAt().toLocalDate().toString(),
                Collectors.summingInt(r -> 1)
            ));
    }

    private List<OutbreakAlert> detectOutbreaks(Map<String, Integer> dailyTrends) {
        List<OutbreakAlert> outbreaks = new ArrayList<>();
        double mean = dailyTrends.values().stream().mapToInt(Integer::intValue).average().orElse(0);
        double stdDev = calculateStandardDeviation(dailyTrends.values(), mean);

        dailyTrends.forEach((date, count) -> {
            if (count > mean + 2 * stdDev) {
                outbreaks.add(new OutbreakAlert(date, count, "HIGH"));
            }
        });
        return outbreaks;
    }

    private List<Hotspot> identifyHotspots(List<PestReport> reports) {
        // Simplified clustering - in production use DBSCAN
        Map<String, List<PestReport>> clusters = new HashMap<>();
        
        for (PestReport report : reports) {
            String gridKey = getGridKey(report.getLatitude(), report.getLongitude());
            clusters.computeIfAbsent(gridKey, k -> new ArrayList<>()).add(report);
        }

        return clusters.entrySet().stream()
            .filter(e -> e.getValue().size() >= 5)
            .map(e -> createHotspot(e.getValue()))
            .collect(Collectors.toList());
    }

    private String getGridKey(Double lat, Double lon) {
        // 10km grid cells
        return ((int)(lat * 10)) + "," + ((int)(lon * 10));
    }

    private Hotspot createHotspot(List<PestReport> reports) {
        double avgLat = reports.stream().mapToDouble(PestReport::getLatitude).average().orElse(0);
        double avgLon = reports.stream().mapToDouble(PestReport::getLongitude).average().orElse(0);
        return new Hotspot(avgLat, avgLon, reports.size(), "HIGH");
    }

    private TrendDirection calculateTrendDirection(Map<String, Integer> dailyTrends) {
        List<Integer> values = new ArrayList<>(dailyTrends.values());
        if (values.size() < 2) return TrendDirection.STABLE;

        double slope = calculateLinearRegression(values);
        if (slope > 0.1) return TrendDirection.INCREASING;
        else if (slope < -0.1) return TrendDirection.DECREASING;
        else return TrendDirection.STABLE;
    }

    private double calculateLinearRegression(List<Integer> values) {
        int n = values.size();
        double sumX = n * (n - 1) / 2.0;
        double sumY = values.stream().mapToDouble(Double::valueOf).sum();
        double sumXY = 0, sumX2 = 0;

        for (int i = 0; i < n; i++) {
            sumXY += i * values.get(i);
            sumX2 += i * i;
        }

        return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    }

    private double calculateStandardDeviation(Collection<Integer> values, double mean) {
        double variance = values.stream()
            .mapToDouble(v -> Math.pow(v - mean, 2))
            .average()
            .orElse(0);
        return Math.sqrt(variance);
    }
}

// ==================== MODEL CLASSES ====================

class PestReport {
    private String reportId;
    private String farmerId;
    private Integer cropId;
    private Double latitude;
    private Double longitude;
    private String description;
    private String imageUrl;
    private Float affectedAreaPercent;
    private ReportStatus status;
    private List<String> detectedPests;
    private Float confidenceScore;
    private SeverityLevel severityLevel;
    private LocalDateTime createdAt;
    private LocalDateTime processedAt;
    private String verifiedBy;
    private LocalDateTime verifiedAt;
    private String verificationNotes;

    // Getters and Setters
    public String getReportId() { return reportId; }
    public void setReportId(String reportId) { this.reportId = reportId; }
    public String getFarmerId() { return farmerId; }
    public void setFarmerId(String farmerId) { this.farmerId = farmerId; }
    public Integer getCropId() { return cropId; }
    public void setCropId(Integer cropId) { this.cropId = cropId; }
    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
    public Float getAffectedAreaPercent() { return affectedAreaPercent; }
    public void setAffectedAreaPercent(Float affectedAreaPercent) { this.affectedAreaPercent = affectedAreaPercent; }
    public ReportStatus getStatus() { return status; }
    public void setStatus(ReportStatus status) { this.status = status; }
    public List<String> getDetectedPests() { return detectedPests; }
    public void setDetectedPests(List<String> detectedPests) { this.detectedPests = detectedPests; }
    public Float getConfidenceScore() { return confidenceScore; }
    public void setConfidenceScore(Float confidenceScore) { this.confidenceScore = confidenceScore; }
    public SeverityLevel getSeverityLevel() { return severityLevel; }
    public void setSeverityLevel(SeverityLevel severityLevel) { this.severityLevel = severityLevel; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getProcessedAt() { return processedAt; }
    public void setProcessedAt(LocalDateTime processedAt) { this.processedAt = processedAt; }
    public String getVerifiedBy() { return verifiedBy; }
    public void setVerifiedBy(String verifiedBy) { this.verifiedBy = verifiedBy; }
    public LocalDateTime getVerifiedAt() { return verifiedAt; }
    public void setVerifiedAt(LocalDateTime verifiedAt) { this.verifiedAt = verifiedAt; }
    public String getVerificationNotes() { return verificationNotes; }
    public void setVerificationNotes(String notes) { this.verificationNotes = notes; }
}

enum ReportStatus { SUBMITTED, VERIFIED, REJECTED, RESOLVED }
enum SeverityLevel { LOW, MEDIUM, HIGH, CRITICAL }
enum TrendDirection { INCREASING, STABLE, DECREASING }

class Alert {
    private String alertId;
    private String farmerId;
    private String reportId;
    private Double distanceKm;
    private LocalDateTime createdAt;
    private Boolean sent;

    public String getAlertId() { return alertId; }
    public void setAlertId(String alertId) { this.alertId = alertId; }
    public String getFarmerId() { return farmerId; }
    public void setFarmerId(String farmerId) { this.farmerId = farmerId; }
    public String getReportId() { return reportId; }
    public void setReportId(String reportId) { this.reportId = reportId; }
    public Double getDistanceKm() { return distanceKm; }
    public void setDistanceKm(Double distanceKm) { this.distanceKm = distanceKm; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public Boolean getSent() { return sent; }
    public void setSent(Boolean sent) { this.sent = sent; }
}

class PestTrendAnalysis {
    private Map<String, Integer> dailyTrends;
    private List<OutbreakAlert> detectedOutbreaks;
    private List<Hotspot> hotspots;
    private TrendDirection trendDirection;

    public Map<String, Integer> getDailyTrends() { return dailyTrends; }
    public void setDailyTrends(Map<String, Integer> dailyTrends) { this.dailyTrends = dailyTrends; }
    public List<OutbreakAlert> getDetectedOutbreaks() { return detectedOutbreaks; }
    public void setDetectedOutbreaks(List<OutbreakAlert> detectedOutbreaks) { this.detectedOutbreaks = detectedOutbreaks; }
    public List<Hotspot> getHotspots() { return hotspots; }
    public void setHotspots(List<Hotspot> hotspots) { this.hotspots = hotspots; }
    public TrendDirection getTrendDirection() { return trendDirection; }
    public void setTrendDirection(TrendDirection trendDirection) { this.trendDirection = trendDirection; }
}

class OutbreakAlert {
    private String date;
    private Integer reportCount;
    private String severity;

    public OutbreakAlert(String date, Integer reportCount, String severity) {
        this.date = date;
        this.reportCount = reportCount;
        this.severity = severity;
    }

    public String getDate() { return date; }
    public Integer getReportCount() { return reportCount; }
    public String getSeverity() { return severity; }
}

class Hotspot {
    private Double latitude;
    private Double longitude;
    private Integer reportCount;
    private String intensity;

    public Hotspot(Double latitude, Double longitude, Integer reportCount, String intensity) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.reportCount = reportCount;
        this.intensity = intensity;
    }

    public Double getLatitude() { return latitude; }
    public Double getLongitude() { return longitude; }
    public Integer getReportCount() { return reportCount; }
    public String getIntensity() { return intensity; }
}

class ControlMeasure {
    private String measureName;
    private String description;
    private Float effectiveness;
    private Float costEstimate;
    private String instructions;

    public String getMeasureName() { return measureName; }
    public String getDescription() { return description; }
    public Float getEffectiveness() { return effectiveness; }
    public Float getCostEstimate() { return costEstimate; }
    public String getInstructions() { return instructions; }
}

class PesticideLog {
    private String logId;
    private String farmerId;
    private String pesticideName;
    private Float quantity;
    private String unit;
    private String notes;
    private LocalDateTime applicationDate;

    public String getLogId() { return logId; }
    public void setLogId(String logId) { this.logId = logId; }
    public String getFarmerId() { return farmerId; }
    public void setFarmerId(String farmerId) { this.farmerId = farmerId; }
    public String getPesticideName() { return pesticideName; }
    public void setPesticideName(String pesticideName) { this.pesticideName = pesticideName; }
    public Float getQuantity() { return quantity; }
    public void setQuantity(Float quantity) { this.quantity = quantity; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
    public LocalDateTime getApplicationDate() { return applicationDate; }
    public void setApplicationDate(LocalDateTime applicationDate) { this.applicationDate = applicationDate; }
}

class Farmer {
    private String farmerId;
    private String name;
    private Double latitude;
    private Double longitude;
    private Boolean alertEnabled;

    public String getFarmerId() { return farmerId; }
    public String getName() { return name; }
    public Double getLatitude() { return latitude; }
    public Double getLongitude() { return longitude; }
    public Boolean isAlertEnabled() { return alertEnabled; }
}
