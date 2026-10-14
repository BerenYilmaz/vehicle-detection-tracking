import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors


class YOLOSpotlight:

    def __init__(self, model="runs/detect/train/weights/best.pt", source="carDetection.mp4", zoom_margin=60,
                 line_start=(500, 400), line_end=(2000, 400)):

        self.model = YOLO(model)
        self.names = self.model.names

        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Error reading video file"

        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))
        self.writer = cv2.VideoWriter("YOLO-spotlight.mp4",
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, (self.w, self.h))

        self.zoom_margin = zoom_margin

        self.line_start = line_start
        self.line_end = line_end
        self.track_side_history = {}
        self.counted_ids = set()
        self.count = 0

        self.window_name = "YOLO Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.current_data = None
        self.selected_id = None
        self.ann = None

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and self.selected_id is None and hasattr(self, "current_data"):
            if self.current_data is None:
                return
            boxes, ids = self.current_data
            self.selected_id = None

            if boxes is not None and ids is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.selected_id = int(ids[i])
                        print(f"Arac kilitlendi -> ID:{self.selected_id}")
                        break

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
                selected_box = boxes[i].numpy()
                zoomed = self.get_zoomed_frame(im0, selected_box)
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
        print("- Bir araci secmek icin uzerine tikla")
        print("- Zoomdan cikmak icin 'z' tusuna bas")
        print("- Cikmak icin 'q' tusuna bas")

        while self.cap.isOpened():
            success, im0 = self.cap.read()

            if not success:
                print("End of video or failed to read image.")
                break

            results = self.model.track(im0, persist=True)
            self.ann = Annotator(im0, line_width=4)

            display_frame = im0

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu()
                    ids = result.boxes.id.cpu()
                    clss = result.boxes.cls.tolist()

                    self.current_data = (boxes, ids)

                    for box, obj_id, cls in zip(boxes, ids.tolist(), clss):
                        self.ann.box_label(box, label=f"{self.names[cls]} ID:{int(obj_id)}",
                                           color=colors(6 if cls == 2 else cls, True))
                        self.update_line_count(box.numpy(), int(obj_id))

                    display_frame = self.process_selected_object(im0, boxes, ids)
                else:
                    self.current_data = None

            cv2.line(im0, self.line_start, self.line_end, (255, 0, 0), 2)
            cv2.putText(im0, f"Arac Sayisi: {self.count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            self.writer.write(im0)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.selected_id = None
                print("Zoomdan cikildi, normal gorunume donuldu.")

        # Cleanup
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

        print(f"Toplam gecen arac sayisi: {self.count}")


if __name__ == "__main__":
    tracker = YOLOSpotlight(
        model="runs/detect/train/weights/best.pt",
        source="carVideo.mp4",
        line_start=(500, 400),
        line_end=(1500, 400)
    )
    tracker.run()
