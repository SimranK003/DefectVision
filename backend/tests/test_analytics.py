def test_analytics_empty_state_has_no_fabricated_data(client):
    response = client.get("/analytics/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 0
    assert body["completed_predictions"] == 0
    assert body["images_with_defects"] == 0
    assert body["defect_rate"] == 0.0
    assert body["defects_by_category"] == {}
    assert body["recent_predictions"] == []
    assert body["batch_stats"]["total_batches"] == 0


def test_analytics_reflects_real_predictions(client, production_model, fake_engine, sample_jpeg_bytes):
    for _ in range(4):
        client.post("/predict", files={"file": ("part.jpg", sample_jpeg_bytes, "image/jpeg")})

    response = client.get("/analytics/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 4
    assert body["completed_predictions"] == 4
    assert body["images_with_defects"] == 4
    assert body["defect_rate"] == 1.0
    assert body["defects_by_category"] == {"crazing": 4}
    assert len(body["recent_predictions"]) == 4
    # fake_engine always returns confidence 0.91 -> falls in the 90-100% bucket
    buckets = {b["range_label"]: b["count"] for b in body["confidence_distribution"]}
    assert buckets["90-100%"] == 4
    assert buckets["0-50%"] == 0
