from paddleocr import PaddleOCR
import cv2
import json
import os
import time
import numpy as np
from sklearn.cluster import DBSCAN

def process_images_paddleocr(image_dir, output_image_dir=None, json_output_path=None):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    results_data = {}

    if output_image_dir and not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, f"paddleocr_{image_name}") if output_image_dir else None

        print(f"Processing PaddleOCR for: {image_name}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_name}. Skipping.")
            continue

        start_time = time.time()
        ocr_result = ocr.ocr(image_path, det=True, cls=True, rec=True)
        end_time = time.time()
        processing_time = end_time - start_time

        file_results = []
        img_display = img.copy()
        flat_results = ocr_result[0] if ocr_result and ocr_result[0] else []
        boxes = []
        for i, (box, content) in enumerate(flat_results):
            text, score = content
            x1, y1 = box[0]
            x2, y2 = box[2]
            w, h = x2 - x1, y2 - y1
            boxes.append([int(x1), int(y1), int(w), int(h), text, score, i])

        clusters = cluster_boxes(boxes=boxes, eps=50)
        cluster_result = []
        for indices_in_cluster in clusters:
            sub_result = [boxes[i] for i in indices_in_cluster]
            if not sub_result:
                continue
            sub_result.sort(key=lambda box: (box[1], box[0]))
            all_x1 = [b[0] for b in sub_result]
            all_y1 = [b[1] for b in sub_result]
            all_x2 = [b[0] + b[2] for b in sub_result]
            all_y2 = [b[1] + b[3] for b in sub_result]
            merged_x1 = min(all_x1)
            merged_y1 = min(all_y1)
            merged_x2 = max(all_x2)
            merged_y2 = max(all_y2)
            merged_w = merged_x2 - merged_x1
            merged_h = merged_y2 - merged_y1
            texts = [b[4] for b in sub_result]
            concatenated_text = " ".join(texts).strip()
            scores = [b[5] for b in sub_result]
            average_score = sum(scores) / len(scores) if scores else 0.0
            cluster_result.append([merged_x1, merged_y1, merged_w, merged_h, concatenated_text, average_score])

        for i, cluster in enumerate(cluster_result):
            top_left = (cluster[0], cluster[1])
            bottom_right = (cluster[0] + cluster[2], cluster[1] + cluster[3])
            if output_image_path:
                cv2.rectangle(img_display, top_left, bottom_right, (0, 255, 0), 2)
                text_label = str(i + 1)
                cv2.putText(img_display, text_label, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            file_results.append({
                "text": f"{i+1}: {cluster[4]}",
                "score": float(cluster[5]),
                "box": [int(top_left[0]), int(top_left[1]), int(cluster[2]), int(cluster[3])]
            })

        if output_image_path:
            cv2.imwrite(output_image_path, img_display)

        results_data[image_name] = {
            "results": file_results,
            "processing_time_ms": processing_time * 1000
        }

    if json_output_path:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)
        print(f"PaddleOCR processing complete. Results saved to {json_output_path}")
    else:
        print("PaddleOCR processing complete.")

    return results_data

def calculate_center(box):
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    return np.array([center_x, center_y])

def cluster_boxes(boxes, eps):
    if not boxes:
        return []
    centers = np.array([calculate_center(box) for box in boxes])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = clustering.labels_
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)
    return list(clusters.values())
