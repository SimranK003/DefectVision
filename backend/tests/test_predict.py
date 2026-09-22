def test_predict_without_production_model_returns_503(client, sample_jpeg_bytes):
    response = client.post("/predict", files={"file": ("part.jpg", sample_jpeg_bytes, "image/jpeg")})

    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()


def test_predict_returns_detections_and_annotated_image(client, production_model, fake_engine, sample_jpeg_bytes):
    response = client.post("/predict", files={"file": ("part.jpg", sample_jpeg_bytes, "image/jpeg")})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["original_filename"] == "part.jpg"
    assert body["model_version"] == "v1"
    assert body["inference_time_ms"] == 12.3
    assert body["annotated_image_url"].startswith("/static/")
    assert len(body["detections"]) == 1
    detection = body["detections"][0]
    assert detection["class_name"] == "crazing"
    assert detection["confidence"] == 0.91
    assert 0.0 <= detection["x_min"] <= 1.0


def test_predict_rejects_invalid_upload_before_touching_the_model(client, production_model, fake_engine):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello world", "text/plain")})

    assert response.status_code == 422


def test_get_prediction_by_id(client, production_model, fake_engine, sample_jpeg_bytes):
    created = client.post("/predict", files={"file": ("part.jpg", sample_jpeg_bytes, "image/jpeg")}).json()

    response = client.get(f"/predictions/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert len(response.json()["detections"]) == 1


def test_get_prediction_404_for_unknown_id(client):
    response = client.get("/predictions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_predictions_paginates_and_summarizes(client, production_model, fake_engine, sample_jpeg_bytes):
    for _ in range(3):
        client.post("/predict", files={"file": ("part.jpg", sample_jpeg_bytes, "image/jpeg")})

    response = client.get("/predictions", params={"limit": 2, "offset": 0})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["items"][0]["top_class"] == "crazing"
    assert body["items"][0]["num_detections"] == 1
