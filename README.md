# Araç Tespiti, Renk Analizi ve Takibi

YOLO tabanlı araç tespiti ve takip uygulamaları. Takip kimlikleri üzerinden çizgi geçişlerini sayar; bir araca tıklandığında yakınlaştırılmış görüntü sunar. İkinci renk analizi sürümü, arka plan etkisini azaltmak için segmentasyon maskelerini kullanır.

## Dosyalar

| Dosya | Amaç |
|---|---|
| `detection.py` | Nesne tespiti, takip, çizgi sayacı ve yakınlaştırma |
| `carDetectionByColor.py` | Sınırlayıcı kutu/gövde kırpımı üzerinde HSV renk filtresi |
| `carDetectionByColorv2.py` | Segmentasyon maskesi, tampon bölgesi ve son 10 gözlemde çoğunluk oylamasıyla renk seçimi |
| `train.ipynb` | `yolo26n.pt` ile araç tespit modeli eğitimi; 100 epoch, 640 görüntü boyutu |
| `trainv2.ipynb` | `yolo26n-seg.pt` ile segmentasyon eğitimi; 100 epoch, 640 görüntü boyutu, batch 16, AMP |

`example.py` boş bir çalışma dosyasıdır.

## Kurulum

Depo kökünde sanal ortamı etkinleştirdikten sonra:

```powershell
python -m pip install -r requirements.txt
jupyter notebook
```

Model adlarını destekleyen bir Ultralytics sürümü gerekir. GPU kullanımı PyTorch/CUDA kurulumuna bağlıdır; eğitim süresi donanıma göre değişir.

## Eğitim verisi

Veri kümelerini aşağıdaki kaynaklardan uygun YOLO biçiminde indirip ilgili klasöre açın. Klasörlerde `data.yaml`, `train/images`, `train/labels`, `valid/images`, `valid/labels` ve varsa test bölümü bulunmalıdır. `data.yaml` yollarını yerel kurulumunuza göre kontrol edin.

| Klasör | Kaynak | Lisans |
|---|---|---|
| `vehicles.v2-release.yolo26/` | [Roboflow 100 Vehicles v2](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2) | CC BY 4.0 |
| `Road-Traffic.v4i.yolo26/` | [Road Traffic v4](https://universe.roboflow.com/mmsseg/road-traffic-rqbit/dataset/4) | CC BY 4.0 |

Road Traffic sınıfları: `Bus`, `Government-car`, `Minibus`, `Motorbike`, `Private-car`, `Taxi`, `Truck`. Segmentasyon için poligon etiketli veri kullanın. Veri kaynağı bilgileri yerel `data.yaml` dosyalarından alınmıştır.

## Çalıştırma

1. İlgili notebook ile eğitim yapın veya uyumlu eğitilmiş ağırlıkları sağlayın.
2. Script sonundaki `model` değerini oluşan `best.pt` dosyasına ayarlayın. Notebook çıktısı otomatik olarak scriptlerin beklediği özel klasöre taşınmaz.
3. `source` değerini kendi videonuza ayarlayın. Varsayılanlar `detection.py` ve ilk renk sürümünde `carVideo.mp4`, ikinci renk sürümünde `carDetection.mp4` dosyasıdır.
4. Çizgi koordinatlarını videonuzun çözünürlüğüne göre ayarlayın.

```powershell
python detection.py
# veya segmentasyon ve renk filtresi:
python carDetectionByColorv2.py
```

İkinci sürümün varsayılan modeli `runs/segment/RoadTraffic/yolo26n_seg/weights/best.pt`, hedef rengi `red` değeridir. `target_color=None` tüm renkleri geçirir. Desteklenen renkler `red`, `yellow`, `green`, `blue`, `black`, `gray`, `white`.

`allowed_classes` modeldeki sınıf adlarıyla birebir eşleşmelidir. İlk renk sürümündeki `['car']`, Road Traffic modelindeki `Private-car` ile eşleşmez; bu modelde `['Private-car']` veya `None` kullanın.

## Kontroller ve çıktı

- Araç kutusuna sol tıklama: takip edilen araca yakınlaştırma.
- `z`: normal görünüme dönme.
- `q`: çıkış ve video yazıcısını kapatma.
- Tespit/ilk renk sürümü: `YOLO-spotlight.mp4`; ikinci sürüm: `YOLO-seg-spotlight.mp4`.
- Sayımlar konsola ve görüntü üzerine yazılır. Aynı çıktı adına yapılan çalıştırmalar dosyanın üzerine yazar.

## Sınırlamalar

Çizgi sayımı, kutu merkezinin çizginin iki tarafı arasında geçmesine dayanır; sonlu çizgi segmentinin sınırlarını kontrol etmez. Takip kimliği değişirse aynı araç tekrar sayılabilir. Renk analizi ışık, gölge ve kamera ayarlarından etkilenir. İkinci sürümde karar çoğunluk oylamasıyla verilir; `color_match_ratio` ve `neutral_match_ratio` parametreleri mevcut karar dalında kullanılmaz. Bu dosya doğrudan çalıştırılmak içindir; sondaki `tracker.run()` ana blok dışında olduğundan modül olarak içe aktarılmaya uygun değildir.
