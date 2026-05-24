package com.example.ungdungkiemphieu.detector

import android.graphics.Bitmap
import android.util.Log
import com.example.ungdungkiemphieu.data.model.BallotDetectionMetadata
import com.example.ungdungkiemphieu.data.model.BallotMarkerMetadata
import com.example.ungdungkiemphieu.data.model.BallotPoint
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.sqrt

class ArUcoDocumentCropper(
    private val minMarkersRequired: Int = 4
) {

    data class CropResult(
        val croppedBitmap: Bitmap?,
        val success: Boolean,
        val usedMarkerIds: List<Int> = emptyList(),
        val corners: List<Point>? = null,
        val hasQR: Boolean = false,
        val qrData: String? = null,
        val totalMarkersDetected: Int = 0,
        val detectionMetadata: BallotDetectionMetadata? = null
    )

    private data class MarkerCandidate(
        val id: Int,
        val corners: List<Point>,
        val center: Point
    )

    private data class ClassifiedMarkers(
        val topRight: MarkerCandidate,
        val bottomRight: MarkerCandidate,
        val bottomLeft: MarkerCandidate
    )

    private val qrAndArUcoDetector = QRAndArUcoDetector()

    suspend fun cropDocument(bitmap: Bitmap): CropResult {
        val detectResult = qrAndArUcoDetector.detectAllMarkers(bitmap)

        Log.d("CROP", "Total markers found: ${detectResult.totalMarkersFound}")
        Log.d("CROP", "QR: ${detectResult.qrData}, ArUco IDs: ${detectResult.arUcoIds.joinToString()}")

        val qrCorners = detectResult.qrCorners
        val qrData = detectResult.qrData
        if (qrCorners == null || qrCorners.size < 4 || qrData.isNullOrBlank()) {
            return CropResult(
                croppedBitmap = null,
                success = false,
                usedMarkerIds = detectResult.arUcoIds,
                hasQR = qrCorners != null,
                qrData = qrData,
                totalMarkersDetected = detectResult.totalMarkersFound
            )
        }

        val markerCandidates = buildMarkerCandidates(detectResult)
        val sharedMarkers = markerCandidates.filter { it.id == SHARED_ARUCO_ID }
        val usableMarkers = if (sharedMarkers.size >= 3) sharedMarkers else markerCandidates

        if (usableMarkers.size < 3 || detectResult.totalMarkersFound < minMarkersRequired) {
            Log.w("CROP", "Not enough reference markers. ArUco=${usableMarkers.size}, total=${detectResult.totalMarkersFound}")
            return CropResult(
                croppedBitmap = null,
                success = false,
                usedMarkerIds = detectResult.arUcoIds,
                hasQR = true,
                qrData = qrData,
                totalMarkersDetected = detectResult.totalMarkersFound
            )
        }

        val qrCenter = averagePoint(qrCorners)
        val selectedMarkers = chooseBestThreeMarkers(usableMarkers, qrCenter)
        val classified = classifyMarkers(selectedMarkers, qrCenter)
        if (classified == null) {
            Log.w("CROP", "Cannot classify ArUco markers into TR/BR/BL")
            return CropResult(
                croppedBitmap = null,
                success = false,
                usedMarkerIds = detectResult.arUcoIds,
                hasQR = true,
                qrData = qrData,
                totalMarkersDetected = detectResult.totalMarkersFound
            )
        }

        val documentCenter = averagePoint(
            listOf(
                qrCenter,
                classified.topRight.center,
                classified.bottomRight.center,
                classified.bottomLeft.center
            )
        )

        val qrInner = closestPoint(qrCorners, documentCenter)
        val topRightInner = closestPoint(classified.topRight.corners, documentCenter)
        val bottomRightInner = closestPoint(classified.bottomRight.corners, documentCenter)
        val bottomLeftInner = closestPoint(classified.bottomLeft.corners, documentCenter)

        var normalizedTopRight = classified.topRight
        var normalizedTopRightInner = topRightInner
        var normalizedBottomLeft = classified.bottomLeft
        var normalizedBottomLeftInner = bottomLeftInner

        var srcPoints = listOf(qrInner, normalizedTopRightInner, bottomRightInner, normalizedBottomLeftInner)
        if (polygonArea(srcPoints) < 0.0) {
            Log.d("CROP", "Reference polygon winding is reversed, swapping top-right and bottom-left")
            normalizedTopRight = classified.bottomLeft
            normalizedTopRightInner = bottomLeftInner
            normalizedBottomLeft = classified.topRight
            normalizedBottomLeftInner = topRightInner
            srcPoints = listOf(qrInner, normalizedTopRightInner, bottomRightInner, normalizedBottomLeftInner)
        }

        if (abs(polygonArea(srcPoints)) < 100.0) {
            Log.w("CROP", "Reference polygon is too small or degenerate")
            return CropResult(
                croppedBitmap = null,
                success = false,
                usedMarkerIds = selectedMarkers.map { it.id },
                hasQR = true,
                qrData = qrData,
                totalMarkersDetected = detectResult.totalMarkersFound
            )
        }

        val metadata = BallotDetectionMetadata(
            qr_raw = qrData,
            ballot_id = parseBallotId(qrData),
            image_width = bitmap.width,
            image_height = bitmap.height,
            src_points = srcPoints.map { it.toBallotPoint() },
            qr_corners = qrCorners.map { it.toBallotPoint() },
            markers = listOf(
                normalizedTopRight.toMetadata("top_right", normalizedTopRightInner),
                classified.bottomRight.toMetadata("bottom_right", bottomRightInner),
                normalizedBottomLeft.toMetadata("bottom_left", normalizedBottomLeftInner)
            ),
            shared_aruco_id = SHARED_ARUCO_ID
        )

        Log.d("CROP", "Client detection metadata ready: src_points=${metadata.src_points}")

        // Keep the original bitmap for upload. The server will use src_points to warp it.
        return CropResult(
            croppedBitmap = bitmap,
            success = true,
            usedMarkerIds = selectedMarkers.map { it.id },
            corners = srcPoints,
            hasQR = true,
            qrData = qrData,
            totalMarkersDetected = detectResult.totalMarkersFound,
            detectionMetadata = metadata
        )
    }

    private fun buildMarkerCandidates(result: QRAndArUcoDetector.DetectionResult): List<MarkerCandidate> {
        val markers = mutableListOf<MarkerCandidate>()
        for (index in result.arUcoIds.indices) {
            val corners = result.arUcoCorners.getOrNull(index)?.toArray()?.toList().orEmpty()
            if (corners.size < 4) continue
            markers.add(
                MarkerCandidate(
                    id = result.arUcoIds[index],
                    corners = corners,
                    center = averagePoint(corners)
                )
            )
        }
        return markers
    }

    private fun chooseBestThreeMarkers(markers: List<MarkerCandidate>, qrCenter: Point): List<MarkerCandidate> {
        if (markers.size <= 3) return markers

        var best = markers.take(3)
        var bestScore = Double.NEGATIVE_INFINITY

        for (i in 0 until markers.size - 2) {
            for (j in i + 1 until markers.size - 1) {
                for (k in j + 1 until markers.size) {
                    val combo = listOf(markers[i], markers[j], markers[k])
                    val score = combo.sumOf { distance(it.center, qrCenter) } +
                        distance(combo[0].center, combo[1].center) +
                        distance(combo[1].center, combo[2].center) +
                        distance(combo[2].center, combo[0].center)
                    if (score > bestScore) {
                        bestScore = score
                        best = combo
                    }
                }
            }
        }

        return best
    }

    private fun classifyMarkers(markers: List<MarkerCandidate>, qrCenter: Point): ClassifiedMarkers? {
        if (markers.size < 3) return null

        val sortedByDistance = markers.sortedBy { distance(it.center, qrCenter) }
        val bottomRight = sortedByDistance.last()
        val adjacent = sortedByDistance.take(2)

        val firstDistance = distance(adjacent[0].center, qrCenter)
        val secondDistance = distance(adjacent[1].center, qrCenter)

        val topRight: MarkerCandidate
        val bottomLeft: MarkerCandidate
        if (firstDistance <= secondDistance) {
            topRight = adjacent[0]
            bottomLeft = adjacent[1]
        } else {
            topRight = adjacent[1]
            bottomLeft = adjacent[0]
        }

        return ClassifiedMarkers(topRight, bottomRight, bottomLeft)
    }

    private fun averagePoint(points: List<Point>): Point {
        val x = points.sumOf { it.x } / points.size
        val y = points.sumOf { it.y } / points.size
        return Point(x, y)
    }

    private fun closestPoint(points: List<Point>, target: Point): Point {
        return points.minByOrNull { distance(it, target) } ?: points.first()
    }

    private fun distance(a: Point, b: Point): Double {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun polygonArea(points: List<Point>): Double {
        var area = 0.0
        for (i in points.indices) {
            val current = points[i]
            val next = points[(i + 1) % points.size]
            area += current.x * next.y - next.x * current.y
        }
        return area / 2.0
    }

    private fun parseBallotId(qrData: String): Int? {
        return qrData.split(":").getOrNull(1)?.toIntOrNull()
    }

    private fun Point.toBallotPoint(): BallotPoint {
        return BallotPoint(x = x, y = y)
    }

    private fun MarkerCandidate.toMetadata(role: String, innerPoint: Point): BallotMarkerMetadata {
        return BallotMarkerMetadata(
            role = role,
            id = id,
            corners = corners.map { it.toBallotPoint() },
            inner_point = innerPoint.toBallotPoint()
        )
    }

    companion object {
        private const val SHARED_ARUCO_ID = 17
    }
}
