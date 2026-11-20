# HVSR Mikrotremor (pengganti Geopsy H/V) — untuk Mac M-series

## 1. Instalasi (sekali saja)

Buka **Terminal**, lalu:

```bash
# install Python 3 dari https://www.python.org/downloads/macos/ (sudah termasuk Tkinter untuk GUI)
cd ~/Downloads/hvsr          # folder tempat hvsr.py disimpan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Setiap kali mau pakai lagi cukup: `cd ~/Downloads/hvsr && source venv/bin/activate`

## 2. Pemakaian

**Mode GUI** (pilih folder lewat jendela):
```bash
python3 hvsr.py
```

Di jendela GUI ada dua tombol:
- **▶ Proses otomatis**: semua titik diproses dengan seleksi STA/LTA.
- **✋ Picking manual**: buka titik terpilih (atau klik dua kali namanya) di jendela picking.

### Picking manual (seperti Geopsy)

Jendela picking menampilkan sinyal Z/N/E di kiri dan kurva H/V di kanan. Setiap window di sinyal
diwarnai sama dengan kurvanya di H/V. Abu-abu muda = window dibuang, hitam = ada gap data.
Kurva dan nilai f0/A0/SESAME langsung dihitung ulang setiap kali kamu mengubah pilihan.

| Aksi | Hasil |
|---|---|
| Klik window di sinyal | pilih / buang window itu |
| Seret di sinyal | buang (atau tambah, sesuai mode di toolbar) semua window di rentang itu |
| Klik sebuah kurva di plot H/V | buang window pemilik kurva itu (kurva menyimpang) |
| Seret kiri-kanan di plot H/V | batasi rentang pencarian f0 (garis biru titik-titik) |
| Tombol Auto STA/LTA / Pilih semua / Kosongkan | atur ulang seleksi |
| Undo atau Ctrl/Cmd+Z | batalkan langkah terakhir |
| 💾 Simpan hasil | simpan JPEG kurva, JPEG sinyal+picking, CSV, laporan, daftar window |

Untuk zoom pakai ikon kaca pembesar di toolbar bawah. Matikan lagi ikonnya sebelum picking.

Picking manual juga bisa dibuka dari Terminal: `python3 hvsr.py ~/Data/TDOM/CLG06 --manual`

**Mode Terminal:**
```bash
python3 hvsr.py ~/Data/TDOM/CLG07            # satu titik
python3 hvsr.py ~/Data/TDOM                  # semua titik (tiap subfolder = 1 titik)
python3 hvsr.py Z.mseed N.mseed E.mseed      # pilih file langsung
```

Opsi penting:

| Opsi | Arti | Default |
|---|---|---|
| `-o FOLDER` | folder hasil | `hasil_hvsr` |
| `--win 40` | panjang window (detik) | 40 |
| `--overlap 50` | overlap window (%) | 0 |
| `--ko 40` | konstanta smoothing Konno-Ohmachi | 40 |
| `--fmin 1 --fmax 15` | batasi pencarian puncak f0 | seluruh kurva |
| `--plot-fmin 0.5 --plot-fmax 20` | rentang frekuensi kurva | 0.5–20 Hz |
| `--reject 2` | buang window yang f0-nya > 2 std dari rata-rata | mati |
| `--no-antitrigger` | matikan seleksi STA/LTA | aktif (STA 1 s, LTA 30 s, 0.2–2.5) |
| `--horizontal geometric` | penggabungan N-E (`squared`/`geometric`/`arithmetic`) | squared (seperti Geopsy) |
| `--dpi 300` | resolusi JPEG | 200 |
| `--show` | tampilkan plot di layar | – |

## 3. Output per titik

- `NAMA_HVSR.jpg` — kurva H/V: garis warna = tiap window, hitam tebal = rata-rata,
  putus-putus = ±std, pita abu-abu = f0 ± std dari window (tampilan seperti Geopsy)
- `NAMA_HVSR.csv` — frekuensi, H/V rata-rata, min, max (bisa dibuka di Excel)
- `NAMA_hasil.txt` — f0, A0, T0, Kg, dan uji 9 kriteria SESAME
- `NAMA_window.csv` — daftar window (waktu mulai/selesai, dipakai atau tidak, f0 per window)
- `NAMA_sinyal_picking.jpg` — tampilan sinyal + window terpilih (hanya dari picking manual)
- `ringkasan_HVSR.csv` — tabel semua titik (jika memproses banyak titik)

## 4. Catatan metode

Detrend → window → anti-trigger STA/LTA (amplitudo absolut) → taper cosinus 5% → FFT →
smoothing Konno-Ohmachi → H = √((N²+E²)/2) → H/V per window → rata-rata & std lognormal.
f0 = puncak kurva rata-rata, A0 = nilai H/V pada f0. Window yang mengandung gap data otomatis dibuang.

Bila kurva punya dua puncak (mis. CLG06), batasi pencarian dengan `--fmin/--fmax` sesuai puncak yang ingin diinterpretasi.
