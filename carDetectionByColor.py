import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors

COLOR_RANGES = {
    "red": [
        ((0, 70, 50), (10, 255, 255)),
        ((165, 70, 50), (180, 255, 255)),
    ],
    "yellow": [((18, 70, 80), (32, 255, 255))],
    "green":  [((40, 60, 40), (85, 255, 255))],
    "blue":   [((95, 70, 40), (135, 255, 255))],

    "black": [((0, 0, 0), (180, 255, 45))],
    "gray":  [((0, 0, 46), (180, 90, 190))],
    "white": [((0, 0, 191), (180, 40, 255))],
}

NEUTRAL_COLORS = {"black", "gray", "white"}


class YOLOSpotlight:

    def __init__(self, model="runs/segment/RoadTraffic/yolo26n_seg/weights/best.pt", source="carVideo.mp4",
                 zoom_margin=60, target_color=None, color_match_ratio=0.5, body_crop_ratio=0.5,
                 line_start=(100, 400), line_end=(900, 400), debug_color=False, allowed_classes=None):

        # Model initialization and classes names
        self.model = YOLO(model)
        self.names = self.model.names

        # Video capturing module
        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Error reading video file"

        # Video writing module
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                               cv2.CAP_PROP_FRAME_HEIGHT,
                                               cv2.CAP_PROP_FPS))
        self.writer = cv2.VideoWriter("YOLO-spotlight.mp4",
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, (self.w, self.h))

        # Zoom settings: kilitlenen aracin etrafinda birakilacak bosluk (piksel)
        self.zoom_margin = zoom_margin

        # Renk filtresi ayarlari: None ise renk filtresi uygulanmaz, tum nesneler gosterilir
        self.target_color = target_color
        self.color_match_ratio = color_match_ratio
        self.body_crop_ratio = body_crop_ratio  # kutunun kac orani (merkeze yakin) kontrol edilecek
        self.debug_color = debug_color  # True ise her kutunun eslesme oranini konsola yazdirir
        self.allowed_classes = allowed_classes  # None ise tum siniflar, ["car"] gibi verilirse sadece o siniflar
        if self.target_color is not None and self.target_color not in COLOR_RANGES:
            raise ValueError(f"Desteklenmeyen renk: {self.target_color}. "
                              f"Secenekler: {list(COLOR_RANGES.keys())}")

        # Cizgi sayaci ayarlari
        self.line_start = line_start
        self.line_end = line_end
        self.track_side_history = {}  # track_id -> cizginin hangi tarafinda oldugu (+/-)
        self.counted_ids = set()      # zaten sayilmis id'ler (ayni araci iki kez saymamak icin)
        self.count = 0

        # Window setup
        self.window_name = "YOLO Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.current_data = None  # for mouse event.
        self.selected_id = None   # tracking state
        self.ann = None           # annotation utils

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



    def is_target_color(self, crop_bgr, debug_label=""):
        if self.target_color is None:
            return True
        if crop_bgr.size == 0:
            return False

        crop_bgr = cv2.medianBlur(crop_bgr, 5)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_RANGES[self.target_color]:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, lower, upper))

        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        matched = cv2.countNonZero(mask_total)
        total = hsv.shape[0] * hsv.shape[1]
        ratio = matched / total if total > 0 else 0

        effective_ratio = 0.28 if self.target_color in NEUTRAL_COLORS else self.color_match_ratio

        if self.debug_color:
            print(f"{debug_label} -> oran: {ratio:.2f}  (esik: {effective_ratio})")

        return ratio >= effective_ratio

    @staticmethod
    def line_side(point, line_start, line_end):
        """Noktanin cizginin hangi tarafinda oldugunu (+/-) doner (cross product)."""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def update_line_count(self, box, track_id):
        """Aracin merkez noktasinin cizgiyi gecip gecmedigini kontrol eder,
        gectiyse ve daha once sayilmadiysa sayaci artirir."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = self.line_side((cx, cy), self.line_start, self.line_end)

        if self.debug_color:
            print(f"  [cizgi] ID:{track_id} merkez=({cx:.0f},{cy:.0f}) taraf={side:.0f}")

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

        # Tam ekran boyutuna buyut (aspect ratio korunmadan, ekrani tam doldursun diye)
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_selected_object(self, im0, boxes, ids):

        if self.selected_id is None:
            return im0

        for i, obj_id in enumerate(ids):
            if int(obj_id) == self.selected_id:
                selected_box = boxes[i]
                zoomed = self.get_zoomed_frame(im0, selected_box)
                if zoomed is not None:
                    cv2.putText(zoomed, f"ZOOM - ID:{self.selected_id}  ('z' ile cik)",
                                (20, zoomed.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                    return zoomed
                return im0

        # Secili arac bu karede tespit edilemedi - kilidi acmadan bekle
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

            raw_frame = im0.copy()  # renk kontrolu icin cizim yapilmamis temiz kare
            h, w = im0.shape[:2]

            # Cizgiyi kutulardan ONCE ciziyoruz ki zum yapildiginda da gorunsun
            cv2.line(im0, self.line_start, self.line_end, (255, 0, 0), 2)
            cv2.putText(im0, f"Arac Sayisi: {self.count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            results = self.model.track(im0, persist=True)  # Object tracking
            self.ann = Annotator(im0, line_width=4)

            display_frame = im0

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu()
                    ids = result.boxes.id.cpu()
                    clss = result.boxes.cls.tolist()

                    # Renk filtresine uyan kutulari ayikla (raw_frame uzerinden, cizimsiz)
                    kept_boxes, kept_ids = [], []
                    for box, obj_id, cls in zip(boxes, ids.tolist(), clss):
                        class_name = self.names[cls]

                        # Sinif filtresi: allowed_classes belirtilmisse sadece o siniflar islenir
                        if self.allowed_classes is not None and class_name not in self.allowed_classes:
                            continue

                        x1, y1, x2, y2 = box.numpy().astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Kutuyu merkeze yakin bir bolgeye daraltarak teker/g�lge/asfalt
                        # gibi araca ait olmayan pikselleri renk kontrolunden disliyoruz
                        bw, bh = x2 - x1, y2 - y1
                        pad_x = int(bw * (1 - self.body_crop_ratio) / 2)
                        pad_y = int(bh * (1 - self.body_crop_ratio) / 2)
                        cx1, cy1 = x1 + pad_x, y1 + pad_y
                        cx2, cy2 = x2 - pad_x, y2 - pad_y

                        # Kutu cok kucukse (kirpma sonrasi sifir/negatif alan olusuyorsa)
                        # daraltmayi uygulamadan orijinal kutuyu kullan
                        if cx2 <= cx1 or cy2 <= cy1:
                            cx1, cy1, cx2, cy2 = x1, y1, x2, y2

                        crop = raw_frame[cy1:cy2, cx1:cx2]

                        if crop.shape[0] < 20 or crop.shape[1] < 20:
                            continue

                        if not self.is_target_color(crop, debug_label=f"ID:{int(obj_id)} sinif:{class_name}"):
                            continue  # hedef renkte degilse bu nesneyi atla

                        kept_boxes.append(box)
                        kept_ids.append(obj_id)

                        self.ann.box_label(box, label=f"{class_name} ID:{int(obj_id)}",
                                           color=colors(6 if cls == 2 else cls, True))
                        self.update_line_count(box.numpy(), int(obj_id))

                    if kept_boxes:
                        boxes_t = np.stack([b.numpy() for b in kept_boxes])
                        ids_t = np.array(kept_ids)
                        self.current_data = (boxes_t, ids_t)  # Store data for mouse callback
                        display_frame = self.process_selected_object(im0, boxes_t, ids_t)
                    else:
                        self.current_data = None
                        self.selected_id = None  # renkte uyan arac kalmadiysa kilidi de birak
                else:
                    self.current_data = None

            self.writer.write(im0)  # kaydedilen video her zaman normal (zumsuz) goruntu
            cv2.imshow(self.window_name, display_frame)  # ekranda seciliyse zumlu goster

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

        color_label = self.target_color if self.target_color else "tum"
        print(f"Toplam gecen {color_label} arac sayisi: {self.count}")


if __name__ == "__main__":
    # Initialize and run tracker
    tracker = YOLOSpotlight(
        model="runs/segment/RoadTraffic/yolo26n_seg/weights/best.pt",
        source="carVideo.mp4",
        target_color=None,     # "red", "blue", "green", "yellow", "white", "black", "gray"
        allowed_classes=["car"], # sadece "car" sinifi isslensin, kamyonlar disarida kalsin
        line_start=(100, 400),  # kendi videonun cozunurlugune gore ayarla
        line_end=(900, 400),
        debug_color=True,
        color_match_ratio=0.50,
        body_crop_ratio=0.50,
    )
    tracker.run()
