package com.example.ungdungkiemphieu.data.model

data class BallotPoint(
    val x: Double,
    val y: Double
)

data class BallotMarkerMetadata(
    val role: String,
    val id: Int? = null,
    val corners: List<BallotPoint>,
    val inner_point: BallotPoint
)

data class BallotDetectionMetadata(
    val version: Int = 1,
    val qr_raw: String? = null,
    val ballot_id: Int? = null,
    val image_width: Int,
    val image_height: Int,
    val src_points: List<BallotPoint>,
    val qr_corners: List<BallotPoint>? = null,
    val markers: List<BallotMarkerMetadata> = emptyList(),
    val shared_aruco_id: Int = 17
)
