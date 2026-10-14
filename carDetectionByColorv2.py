import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import deque, Counter

COLOR_RANGES = {
    "red": [
        ((0, 60, 50), (9, 255, 255)),
        ((140, 60, 50), (180, 255, 255)),
    ],
    "yellow": [((18, 70, 80), (32, 255, 255))],
    "green":  [((40, 60, 40), (85, 255, 255))],
    "blue":   [((95, 70, 40), (135, 255, 255))],
    "black":  [((0, 0, 0), (180, 90, 65))],
    "gray":   [((0, 0, 66), (180, 90, 200))],
    "white":  [((0, 0, 201), (180, 90, 255))],
}


NEUTRAL_COLORS = {"black", "gray", "white"}


class YOLOSegSpotlight:

    def __init__(self, model, source, target_color=None,
                 color_match_ratio=0.5, neutral_match_ratio=0.3,
                 zoom_margin=60, line_start=(100, 400), line_end=(900, 400),
                 debug_color=False, allowed_classes=None, min_mask_pixels=60):



        self.track_color_history = {}
        self.smoothing_window = 10
        self.model = YOLO(model)
        self.names = self.model.names

        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Video acilamadi"

        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        self.writer = cv2.VideoWriter("YOLO-seg-spotlight.mp4",
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, (self.w, self.h))

        self.target_color = target_color
        self.color_match_ratio = color_match_ratio
        self.neutral_match_ratio = neutral_match_ratio
        self.zoom_margin = zoom_margin
        self.debug_color = debug_color
        self.allowed_classes = allowed_classes
        self.min_mask_pixels = min_mask_pixels

        if self.target_color is not None and self.target_color not in COLOR_RANGES:
            raise ValueError(f"Desteklenmeyen renk: {self.target_color}. "
                              f"Secenekler: {list(COLOR_RANGES.keys())}")

        self.line_start = line_start
        self.line_end = line_end
        self.track_side_history = {}
        self.counted_ids = set()
        self.count = 0

        self.window_name = "YOLO Segmentation Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.current_data = None
        self.selected_id = None

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.current_data is None:
            return
        boxes, ids = self.current_data
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_id = int(ids[i])
                print(f"Arac kilitlendi -> ID:{self.selected_id}")
                break

    def extract_bumper_mask(self, mask_crop, bumper_ratio=0.2, horizontal_margin=0.15):
        hh, ww = mask_crop.shape[:2]
        if hh == 0 or ww == 0:
            return mask_crop

        y_start = int(hh * (1 - bumper_ratio))
        x_start = int(ww * horizontal_margin)
        x_end = int(ww * (1 - horizontal_margin))

        bumper_mask = np.zeros_like(mask_crop)
        bumper_mask[y_start:hh, x_start:x_end] = mask_crop[y_start:hh, x_start:x_end]
        return bumper_mask


    def get_dominant_color(self, crop_bgr, mask_crop):
        if crop_bgr.size == 0 or mask_crop.size == 0:
            return None, 0.0, {}

        car_pixel_count = cv2.countNonZero(mask_crop)
        if car_pixel_count < self.min_mask_pixels:
            return None, 0.0, {}

        blurred = cv2.medianBlur(crop_bgr, 5)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        ratios = {}
        for color_name, ranges in COLOR_RANGES.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_crop)

            matched = cv2.countNonZero(color_mask)
            ratios[color_name] = matched / car_pixel_count

        dominant_color = max(ratios, key=ratios.get)
        return dominant_color, ratios[dominant_color], ratios


    def is_target_color(self, crop_bgr, mask_crop, track_id, debug_label=""):
        if self.target_color is None:
            return True
        if crop_bgr.size == 0 or mask_crop.size == 0:
            return False

        bumper_mask = self.extract_bumper_mask(mask_crop)
        bumper_pixel_count = cv2.countNonZero(bumper_mask)

        effective_mask = bumper_mask if bumper_pixel_count >= self.min_mask_pixels else mask_crop

        dominant_color, dominant_ratio, ratios = self.get_dominant_color(crop_bgr, effective_mask)
        if dominant_color is None:
            if self.debug_color:
                print(f"{debug_label} -> maske cok kucuk, atlandi")
            return False

        if track_id not in self.track_color_history:
            self.track_color_history[track_id] = deque(maxlen=self.smoothing_window)
        self.track_color_history[track_id].append(dominant_color)

        history = self.track_color_history[track_id]
        most_common_color, vote_count = Counter(history).most_common(1)[0]
        vote_ratio = vote_count / len(history)

        if self.debug_color:
            detay = "  ".join(f"{k}:{v:.2f}" for k, v in ratios.items())
            kaynak = "tampon" if bumper_pixel_count >= self.min_mask_pixels else "tum_govde"
            print(f"{debug_label} -> [{kaynak}] bu kare: {dominant_color} ({dominant_ratio:.2f})  [{detay}]  "
                f"son {len(history)} kare cogunluk: {most_common_color} ({vote_ratio:.2f})")

        return most_common_color == self.target_color and vote_ratio >= 0.5

    @staticmethod
    def line_side(point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def update_line_count(self, box, track_id):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = self.line_side((cx, cy), self.line_start, self.line_end)

        if track_id in self.track_side_history:
            prev_side = self.track_side_history[track_id]
            if prev_side * side < 0 and track_id not in self.counted_ids:
                self.count += 1
                self.counted_ids.add(track_id)
                print(f"  >>> ID:{track_id} CIZGIYI GECTI! Yeni sayac: {self.count}")

        self.track_side_history[track_id] = side

    def get_zoomed_frame(self, im0, box):
        h, w = im0.shape[:2]
        x1, y1, x2, y2 = box.astype(int)

        zx1 = max(0, x1 - self.zoom_margin)
        zy1 = max(0, y1 - self.zoom_margin)
        zx2 = min(w, x2 + self.zoom_margin)
        zy2 = min(h, y2 + self.zoom_margin)

        crop = im0[zy1:zy2, zx1:zx2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_selected_object(self, im0, boxes, ids):
        if self.selected_id is None:
            return im0

        for i, obj_id in enumerate(ids):
            if int(obj_id) == self.selected_id:
                zoomed = self.get_zoomed_frame(im0, boxes[i])
                if zoomed is not None:
                    cv2.putText(zoomed, f"ZOOM - ID:{self.selected_id}  ('z' ile cik)",
                                (20, zoomed.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                    return zoomed
                return im0

        cv2.putText(im0, f"ID:{self.selected_id} su an gorunmuyor ('z' ile cik)",
                    (20, im0.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        return im0

    def run(self):
        print("- Bir araci secmek icin uzerine tikla (zoom acilir)")
        print("- Zoomdan cikmak icin 'z' tusuna bas")
        print("- Cikmak icin 'q' tusuna bas")

        while self.cap.isOpened():
            success, im0 = self.cap.read()
            if not success:
                print("Video bitti veya okunamadi.")
                break

            raw_frame = im0.copy()
            h, w = im0.shape[:2]

            cv2.line(im0, self.line_start, self.line_end, (255, 0, 0), 2)
            cv2.putText(im0, f"Arac Sayisi: {self.count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            results = self.model.track(im0, persist=True, retina_masks=True, imgsz=960, conf=0.2)
            display_frame = im0

            if results and len(results) > 0:
                result = results[0]

                has_data = (result.boxes is not None and result.boxes.id is not None
                            and result.masks is not None)

                if has_data:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().tolist()
                    clss = result.boxes.cls.tolist()
                    masks = result.masks.data.cpu().numpy()

                    kept_boxes, kept_ids = [], []

                    for i, (box, obj_id, cls) in enumerate(zip(boxes, ids, clss)):
                        class_name = self.names[int(cls)]
                        if self.allowed_classes is not None and class_name not in self.allowed_classes:
                            continue

                        x1, y1, x2, y2 = box.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        mask_full = masks[i]
                        if mask_full.shape[:2] != (h, w):
                            mask_full = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_full = (mask_full > 0.5).astype(np.uint8) * 255

                        crop_bgr = raw_frame[y1:y2, x1:x2]
                        mask_crop = mask_full[y1:y2, x1:x2]

                        if not self.is_target_color(crop_bgr, mask_crop, int(obj_id),
                            debug_label=f"ID:{int(obj_id)} sinif:{class_name}"):
                            continue

                        kept_boxes.append(box)
                        kept_ids.append(obj_id)


                        contours, _ = cv2.findContours(mask_full[y1:y2, x1:x2],
                                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        col = colors(int(cls), True)
                        for c in contours:
                            c[:, :, 0] += x1
                            c[:, :, 1] += y1
                        cv2.drawContours(im0, contours, -1, col, 2)
                        cv2.rectangle(im0, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(im0, f"{class_name} ID:{int(obj_id)}",
                                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                        self.update_line_count(box, int(obj_id))

                    active_ids = set(int(i) for i in ids)
                    for old_id in list(self.track_color_history.keys()):
                        if old_id not in active_ids:
                            del self.track_color_history[old_id]

                    if kept_boxes:
                        boxes_t = np.stack(kept_boxes)
                    if kept_boxes:
                        boxes_t = np.stack(kept_boxes)
                        ids_t = np.array(kept_ids)
                        self.current_data = (boxes_t, ids_t)
                        display_frame = self.process_selected_object(im0, boxes_t, ids_t)
                    else:
                        self.current_data = None
                        self.selected_id = None
                else:
                    self.current_data = None

            self.writer.write(im0)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.selected_id = None
                print("Zoomdan cikildi.")

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

        color_label = self.target_color if self.target_color else "tum"
        print(f"Toplam gecen {color_label} arac sayisi: {self.count}")


if __name__ == "__main__":
    tracker = YOLOSegSpotlight(
        model="runs/segment/RoadTraffic/yolo26n_seg/weights/best.pt",
        source="carDetection.mp4",
        target_color="red",
        allowed_classes=None,
        debug_color=True,
    )
tracker.run()
