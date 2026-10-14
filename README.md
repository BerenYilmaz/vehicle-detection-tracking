# Ara� Tespiti, Renk Analizi ve Takibi

YOLO tabanl� ara� tespiti ve takip uygulamalar�. Takip kimlikleri �zerinden �izgi ge�i�lerini sayar; bir araca t�kland���nda yak�nla�t�r�lm�� g�r�nt� sunar. �kinci renk analizi s�r�m�, arka plan etkisini azaltmak i�in segmentasyon maskelerini kullan�r.

## Dosyalar

| Dosya | Ama� |
|---|---|
| `detection.py` | Nesne tespiti, takip, �izgi sayac� ve yak�nla�t�rma |
| `carDetectionByColor.py` | S�n�rlay�c� kutu/g�vde k�rp�m� �zerinde HSV renk filtresi |
| `carDetectionByColorv2.py` | Segmentasyon maskesi, tampon b�lgesi ve son 10 g�zlemde �o�unluk oylamas�yla renk se�imi |
| `train.ipynb` | `yolo26n.pt` ile ara� tespit modeli e�itimi; 100 epoch, 640 g�r�nt� boyutu |
| `trainv2.ipynb` | `yolo26n-seg.pt` ile segmentasyon e�itimi; 100 epoch, 640 g�r�nt� boyutu, batch 16, AMP |

`example.py` bo� bir �al��ma dosyas�d�r.

## Kurulum

Depo k�k�nde sanal ortam� etkinle�tirdikten sonra:

```powershell
python -m pip install -r requirements.txt
jupyter notebook
```

Model adlar�n� destekleyen bir Ultralytics s�r�m� gerekir. GPU kullan�m� PyTorch/CUDA kurulumuna ba�l�d�r; e�itim s�resi donan�ma g�re de�i�ir.

## E�itim verisi

Veri k�melerini a�a��daki kaynaklardan uygun YOLO bi�iminde indirip ilgili klas�re a��n. Klas�rlerde `data.yaml`, `train/images`, `train/labels`, `valid/images`, `valid/labels` ve varsa test b�l�m� bulunmal�d�r. `data.yaml` yollar�n� yerel kurulumunuza g�re kontrol edin.

| Klas�r | Kaynak | Lisans |
|---|---|---|
| `vehicles.v2-release.yolo26/` | [Roboflow 100 Vehicles v2](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2) | CC BY 4.0 |
| `Road-Traffic.v4i.yolo26/` | [Road Traffic v4](https://universe.roboflow.com/mmsseg/road-traffic-rqbit/dataset/4) | CC BY 4.0 |

Road Traffic s�n�flar�: `Bus`, `Government-car`, `Minibus`, `Motorbike`, `Private-car`, `Taxi`, `Truck`. Segmentasyon i�in poligon etiketli veri kullan�n. Veri kayna�� bilgileri yerel `data.yaml` dosyalar�ndan al�nm��t�r.

## �al��t�rma

1. �lgili notebook ile e�itim yap�n veya uyumlu e�itilmi� a��rl�klar� sa�lay�n.
2. Script sonundaki `model` de�erini olu�an `best.pt` dosyas�na ayarlay�n. Notebook ��kt�s� otomatik olarak scriptlerin bekledi�i �zel klas�re ta��nmaz.
3. `source` de�erini kendi videonuza ayarlay�n. Varsay�lanlar `detection.py` ve ilk renk s�r�m�nde `carVideo.mp4`, ikinci renk s�r�m�nde `carDetection.mp4` dosyas�d�r.
4. �izgi koordinatlar�n� videonuzun ��z�n�rl���ne g�re ayarlay�n.

```powershell
python detection.py
# veya segmentasyon ve renk filtresi:
python carDetectionByColorv2.py
```

�kinci s�r�m�n varsay�lan modeli `runs/segment/RoadTraffic/yolo26n_seg/weights/best.pt`, hedef rengi `red` de�eridir. `target_color=None` t�m renkleri ge�irir. Desteklenen renkler `red`, `yellow`, `green`, `blue`, `black`, `gray`, `white`.

`allowed_classes` modeldeki s�n�f adlar�yla birebir e�le�melidir. �lk renk s�r�m�ndeki `['car']`, Road Traffic modelindeki `Private-car` ile e�le�mez; bu modelde `['Private-car']` veya `None` kullan�n.

## Kontroller ve ��kt�

- Ara� kutusuna sol t�klama: takip edilen araca yak�nla�t�rma.
- `z`: normal g�r�n�me d�nme.
- `q`: ��k�� ve video yaz�c�s�n� kapatma.
- Tespit/ilk renk s�r�m�: `YOLO-spotlight.mp4`; ikinci s�r�m: `YOLO-seg-spotlight.mp4`.
- Say�mlar konsola ve g�r�nt� �zerine yaz�l�r. Ayn� ��kt� ad�na yap�lan �al��t�rmalar dosyan�n �zerine yazar.

## S�n�rlamalar

�izgi say�m�, kutu merkezinin �izginin iki taraf� aras�nda ge�mesine dayan�r; sonlu �izgi segmentinin s�n�rlar�n� kontrol etmez. Takip kimli�i de�i�irse ayn� ara� tekrar say�labilir. Renk analizi ���k, g�lge ve kamera ayarlar�ndan etkilenir. �kinci s�r�mde karar �o�unluk oylamas�yla verilir; `color_match_ratio` ve `neutral_match_ratio` parametreleri mevcut karar dal�nda kullan�lmaz. Bu dosya do�rudan �al��t�r�lmak i�indir; sondaki `tracker.run()` ana blok d���nda oldu�undan mod�l olarak i�e aktar�lmaya uygun de�ildir.
