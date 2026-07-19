# DR-Grade Self-XAI (Desktop)

Aplikasi desktop (PySide6) untuk **grading diabetic retinopathy** dari citra
fundus menggunakan **Concept Bottleneck Model (CBM)**, dilengkapi penjelasan
**Self-XAI** yang faithful berbasis konsep lesi yang benar-benar terdeteksi
model + RAG dari knowledge base klinis.

## Alur

```
Fundus image
   → preprocessing (CLAHE + resize 512 + normalize, sama seperti training)
   → CBM (strict) → 6 konsep lesi (MA/HE/EX/SE/IRMA/NV) → grade DR 0–4
   → U-Net (resnet34) → segmentasi 6 lesi → overlay berwarna di atas citra
   → RAG deterministik (fgadr_concept_kb.yaml) untuk konsep yang aktif
   → LLM Gemini (faithful, tidak menambah/hapus lesi) → narasi klinis
      (fallback ke template KB bila API gagal/offline/tanpa key)
```

Overlay lesi bisa di-toggle per lesi (MA/HE/EX/SE/IRMA/NV) dengan legenda berwarna,
dan ambang (threshold) overlay bisa diubah live tanpa inferensi ulang.

## Tampilan (dashboard "Reti-AI", tata letak 3-zona)

GUI bertema gelap dengan hierarki baca yang jelas (mengikuti kritik UX):

- **Patient banner (sticky, full-width)** di paling atas: nama besar, ID, umur/gender,
  DOB, + badge status sesi. Selalu terlihat (konfirmasi identitas = langkah paling
  kritis di alur klinis).
- **Zona INPUT** (kiri): 3 panel fundus — (1) citra asli, (2) deteksi lesi (bounding
  box dari komponen mask U-Net), (3) peta segmentasi (lesi berwarna di latar hitam) —
  + kontrol overlay per lesi.
- **Zona ANALYSIS** (tengah): *KEY DRIVERS* sebagai concept chips berwarna;
  grafik **Grade Distribution** dengan **ramp sekuensial** (kuning→oranye→merah),
  label grade + %, dan **hatching** untuk grade berat (aman color-blind); panel
  **Quantitative** (progressive disclosure — toggle "Show details"), dihitung
  **nyata** dari mask (jumlah lesi, luas piksel, % area retina, lokasi). Tanpa mm².
- **Zona VERDICT** (kanan, paling menonjol — aksen atas berwarna): badge severity
  **bertint** (bukan teks merah polos), diagnosis, badge confidence, dan kotak
  **Recommended Action**. Warna **merah hanya** dipakai untuk *callout rujukan urgen*
  (grade ≥ 3) — sesuai prinsip semantic color.
- **Narrative (LLM+RAG), full-width bawah**: struktur 3-tier — *Clinical Implication*
  ringkas + tombol *Read full explanation* (collapsible) untuk prosa lengkap
  (summary/evidence/interpretation/limitations/caveats).

- **Generate Report** (di kolom Verdict, bawah narasi): ekspor **PDF** (berisi
  data pasien, klasifikasi, 3 citra fundus, konsep, distribusi grade, analisis
  kuantitatif, narasi + disclaimer) atau **JSON** — dari hasil nyata. Pakai
  QTextDocument/QPrinter bawaan Qt (tanpa dependensi tambahan).

Catatan kejujuran data: header pasien adalah **placeholder demo** (bisa diedit),
grafik "Progression Over Time" & klaim "similar cases" dari mockup **tidak**
disertakan karena tidak ada data pendukung.

Prinsip **faithfulness**: narator hanya menjelaskan konsep yang difire model dan
**tidak** boleh mengubah grade — knowledge base hanya menyuplai pengetahuan
deskriptif/klinis, bukan menentukan lesi.

## Struktur

| File | Fungsi |
|------|--------|
| `main.py` | GUI PySide6 (entry point) |
| `inference.py` | Load checkpoint + preprocessing + prediksi |
| `model_def.py` | Definisi arsitektur CBM (ekstraksi bersih dari `fusion.py`, tanpa efek samping) |
| `segmentation.py` | Load U-Net (smp) + prediksi mask lesi + overlay berwarna |
| `xai.py` | RAG deterministik + konteks KB + narator template (fallback) |
| `llm_gemini.py` | Narator LLM Gemini (faithful) + loader API key aman |
| `_smoketest.py` | Uji backend tanpa GUI |

Model & KB dibaca dari folder induk: `../best_strict_cbm.pt` dan
`../fgadr_concept_kb.yaml`.

## Menjalankan

```powershell
pip install -r requirements.txt   # (torch sesuai CUDA/CPU Anda)
cd app
python main.py
```

Uji backend saja (tanpa GUI):

```powershell
python _smoketest.py
```

## Penjelasan LLM (Gemini) & keamanan API key

Narasi dibuat oleh **Gemini** (`gemini-3.1-flash-lite-preview`) dengan pola
*faithful*: LLM **tidak** menentukan/menambah/menghapus lesi — ia hanya menjelaskan
konsep yang sudah di-FIRED oleh CBM, memakai kutipan yang di-retrieve deterministik
dari `fgadr_concept_kb.yaml`. Bila API gagal/offline/tanpa key → otomatis fallback
ke narator template (offline).

**API key — aman untuk repo publik (GitHub):**

1. Env var `GEMINI_API_KEY` (diutamakan), lalu
2. file lokal `gemini_config.json` yang **sudah masuk `.gitignore`** (tidak ikut commit).
3. Template `gemini_config.example.json` boleh di-commit (tanpa key).

Key **tidak pernah** di-hard-code di source. Sebelum `git push`, pastikan
`gemini_config.json` benar-benar ter-ignore (`git status` tidak menampilkannya).

> ⚠️ Ganti/rotate API key Anda bila pernah terkirim sebagai teks biasa (chat/log).
> Model file `*.pt` besar (>50 MB) — pertimbangkan Git LFS atau exclude dari repo.

## Catatan / rencana lanjutan

- Narasi memakai **Gemini** dengan fallback template lokal. Bahasa keluaran: English.
- Overlay segmentasi lesi (U-Net) sudah termasuk. Ekspor laporan PDF/JSON belum —
  bisa ditambahkan sebagai fase berikutnya.
- Ini alat **decision support**, bukan diagnosis. Perlu konfirmasi oftalmologis.
