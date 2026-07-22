# Reproduksi PosNegDM: Reinforced Sequential Decision-Making for Sepsis Treatment

Reproduksi paper "Reinforced Sequential Decision-Making for Sepsis Treatment: The
PosNegDM Framework with Mortality Classifier and Transformer" (Tamboli et al.,
IEEE JBHI 2024) dari awal menggunakan dataset MIMIC-III v1.4.

## Ringkasan Pipeline

Paper ini menggunakan pipeline 3 repo yang berbeda secara berantai sebelum
sampai ke training PosNegDM:

1.  **AI_Clinician (Komorowski et al., Nature 2018)** -- Ekstraksi data sepsis
    dari MIMIC-III mentah via SQL + MATLAB, menghasilkan dataset mentah 48
    jam pertama per pasien.
2.  **sepsisrl (Raghu et al., 2017)** -- Preprocessing Python: diskretisasi
    intervensi medis, split train/val/test, normalisasi dan scaling fitur.
3.  **PosNegDM (Tamboli et al., 2024)** -- Training Decision Transformer
    dengan mortality classifier adversarial.

Proses reproduce mencatat setiap kendala, perbedaan environment, dan
penyesuaian yang diperlukan.

## Spesifikasi Sistem

- CPU: AMD Ryzen 7 5700X
- GPU: NVIDIA GeForce RTX 5060 8GB (Blackwell, sm_120)
- RAM: 32 GB DDR4
- OS: Manjaro Linux (KDE)
- Database: PostgreSQL 16, MIMIC-III v1.4, schema mimiciii

## Langkah Reproduksi

### 1. Ekstraksi SQL (Komorowski)

Dataset: MIMIC-III v1.4 (paper asli menggunakan v1.3).

Notebook `AIClinician_Data_extract_MIMIC3_140219.ipynb` menjalankan query SQL
ke database `mimic` (PostgreSQL) untuk mengekstrak data pasien sepsis.
Output berupa CSV di `data/extracted/`.

Perbedaan dengan paper:
- Paper menggunakan MIMIC-III v1.3, reproduce menggunakan v1.4
- Jumlah data lebih banyak: 20.948 ICU stays (v1.4) vs 17.083 ICU
  admissions (v1.3, per paper Komorowski). Sekitar 3.865 lebih banyak.
  Catatan: paper Komorowski menggunakan ICU admission sebagai unit
  sedangkan pipeline ini menggunakan ICU stay -- keduanya tidak persis sama.
- Total baris data: 278.615 (v1.4) vs ~246.000 (v1.3 paper Komorowski).

Environment: Python 2.7.18 (preproc_venv) dengan numpy 1.16.6, pandas 0.24.2.

### 2. Definisi Kohort Sepsis (Komorowski, MATLAB)

Script `AIClinician_sepsis3_def_160219.m` mendefinisikan kohort sepsis-3
berdasarkan kriteria dari paper Komorowski. Menghasilkan `sepsis_mimiciii.csv`
(21.322 baris).

### 3. Dataset Building (Komorowski, MATLAB)

Script `AIClinician_mimic3_dataset_160219.m` membangun dataset 48 jam pertama
dengan fitur vital, laboratorium, intervensi, dan skor SOFA/SIRS.

Output: `raw/MKdataset07Feb17.csv` -- 278.615 baris, 59 kolom, 20.948 unique
icustayid.

### 4. Diskretisasi Intervensi (Raghu, Python 2.7)

Notebook `process_interventions.ipynb`:
- Menghitung kuartil untuk intervensi medis (IV fluids, vasopressors)
- Mendiskretisasi action menjadi 5 level vaso x 5 level IV = 25 kombinasi
- Output: `processed/discretised_input_data.csv`

**Kendala/Catatan:**
- Nama kolom `input_4hourly_tev` di notebook tidak ada di data (hanya
  `input_4hourly`) -- diperbaiki.

### 5. Preprocessing Fitur (Raghu, Python 2.7)

Notebook `preprocess_data.ipynb`:
- Menambahkan kolom reward (-100 untuk meninggal, +100 untuk survive,
  0 untuk intermediate step)
- Split data: 70% train, 10% val, 20% test
- Normalisasi: binary fields (dipusatkan ke -0.5/0.5), z-score untuk
  distribusi normal, log-transform untuk skewed features
- Min-max scaling ke [0,1]
- Output 6 file: `rl_{train,val,test}_set_{un,}scaled.csv`

**Kendala/Catatan:**
- `capping_values.csv` dihapus karena outlier sudah di-handle oleh MATLAB
  (NaN + KNN imputation). Operasi Komorowski set outlier ke NaN (bukan clip),
  jadi capping redundant.

### 6. Bridge ke Format PosNegDM (Python 3)

Script `data/preprocessing/bridge_to_posnegdm.py`:
- Menggabungkan 3 scaled CSVs
- Renumber icustayid ke traj ID sequential
- Menghitung step per trajectory
- Menggabungkan action: a = vaso_input * 5 + iv_input (0-24)
- Menskalakan reward: dari +/-100 ke +/-1 (karena source code PosNegDM
  menggunakan threshold `== 1` untuk positive trajectory)
- Memilih 47 kolom observasi sesuai spesifikasi read_dst di utils.py
- Output: `final/sepsis_final_data_withTimes.csv` (278.615 baris, 53 kolom)
- Output: `final/observations.json` (untuk mortality classifier)

**Kendala/Catatan:**
- Repo dari posnegdm tidak memberikan kode sehingga file ini dibuat sendiri beredasarkan informasi dari utils.py (fungsi read_dst()), train.py, dan sample_data.csv karena itu satu-satunya petunjuk format data yang ada di repo mereka.

### 7. Training PosNegDM (Python 3)

**Environment:**
- Tidak bisa menggunakan PyTorch 2.3.0 seperti paper karena RTX 5060
  (Blackwell sm_120) tidak didukung.
- Menggunakan Python 3.14 (myenv) dengan PyTorch 2.12.1+cu130 (dari env
  yang sudah ada).

**Kendala Environment:**
- Transformers versi baru (4.53.x) menghapus `SequenceSummary` dari
  `modeling_utils`. Import ini tidak dipakai di kode -- dihapus.
- Argumen `--batch_size` hanya mempengaruhi training batch, bukan test batch.
  `get_test_batch()` memiliki default `batch_size=563` yang menyebabkan OOM
  pada GPU 8GB saat `test_complete_traj`. Diturunkan ke 128.

**Hasil Training (20 iterasi):**

| Metrik | Nilai |
|--------|-------|
| Training loss final | 2.716 |
| Training accuracy final | 94.3% |
| Test accuracy (mode=0) | 89.5% |
| Pos test accuracy (mode=1) | 92.0% |
| Neg test accuracy (mode=-1) | 85.3% |
| State loss final | 0.085 |
| Time per iterasi | ~104 detik |
| Total waktu | ~35 menit |

## Perbedaan dengan Paper Asli

### Data
| Aspek | Paper (Tamboli et al.) | Reproduce |
|-------|----------------------|---------------|
| MIMIC version | v1.3 | v1.4 |
| ICU stays / admissions | Paper Komorowski: 17.083 ICU admissions | 20.948 ICU stays |
| - | (perbedaan karena v1.4 memiliki lebih banyak | data dibanding v1.3, serta perbedaan definisi |
| - | admission vs stay) | |
| Baris data | Paper Komorowski: ~246.000 baris | 278.615 baris |
| Preprocessing pipeline | Menggunakan pipeline internal | Pipeline Komorowski + Raghu |
| Reward scaling | +/-1 (dari kode) | +/-100 dari Raghu, di-scale ke +/-1 |
| Reward structure | Reward per step (tidak diketahui detailnya) | Reward 0 di setiap intermediate step,
| | | +100/-100 hanya di step terakhir |
| Split | Tidak disebut | 70/10/20 (Raghu) |

### Environment
| Aspek | Paper | Reproduce |
|-------|-------|-----------|
| Python | 3.10 (conda) | 3.14 (myenv) |
| PyTorch | 2.3.0 | 2.12.1 |
| GPU | Tidak disebut | RTX 5060 8GB |
| CUDA | Tidak disebut | 13.3 (driver) |

### Source Code
| Issue | Paper | Reproduce |
|-------|-------|-----------|
| Positive detection | `rewards[-1] == 1` | Data punya reward 100, harus di-scale |
| Transformers | 4.32.1 | 4.53.3 (SequenceSummary dihapus) |
| Test batch size | 563 (hardcoded) | 128 (OOM pada 8GB) |

## Perbandingan Hasil dengan Paper

### Tabel Perbandingan

| Metrik | Paper (POSNEGDM) | Reproduce |
|--------|:-:|:-:|
| Action Accuracy (Positive Test) | 94.6% | 92.0% |
| Action Accuracy (Test, all) | - | 89.5% |
| Survival Rate | **97.39%** (dari step-by-step) | 99.90%* |
| Mortality Rate (Step-by-step) | 2.61% | 0.10%* |
| Mortality Rate (Complete Trajectory) | 0.18% | 0.29%* |
| Mortality Classifier Accuracy | 96.7% | 96.7% (pretrained) |
| Dataset trajectories | 19.614 | 20.948 |
| Hyperparameter (alpha, beta, gamma) | (1, 0.1, 1) | (1, 0.8, 1) (karena di source code pakai ini dan ga diubah) |

Catatan:
- (*) Mortality rate pada reproduce dihitung menggunakan mortality
  classifier `NNfinal.pth` yang sama dengan paper, tapi classifier ini
  dilatih dengan data preprocessing versi penulis (bukan pipeline
  Komorowski + Raghu). Karena distribusi fitur data kita berbeda (min-max
  scaling [0,1] vs preprocessing versi penulis), hasil mortality rate
  tidak dapat dipercaya dan tidak bisa dibandingkan secara langsung
  dengan paper.
- Beta=0.8 di reproduce (hardcoded di seq_trainer.py), sedangkan paper
  melaporkan hasil utama dengan beta=0.1.
- Dataset reproduce lebih besar (20.948 vs 19.614 trajectories) karena
  menggunakan MIMIC-III v1.4 yang memiliki lebih banyak data daripada v1.3.
- Action accuracy reproduce lebih rendah kemungkinan karena perbedaan
  pipeline preprocessing (fitur scaling, reward structure) dan dataset.

Paper melaporkan hasil pada 3 metrik: survival rate, length of stay, dan accuracy pembedaan positif/negatif. Dataset dan preprocessing yang digunakan paper tidak persis sama dengan reproduce ini, sehingga perbandingan langsung tidak dapat dilakukan.

Yang dapat diverifikasi:
- Model konvergen: loss turun dari 5.613 ke 2.716
- Accuracy naik dari random (~4%) ke 89.5%
- State prediction error menurun (state loss: 2.464 -> 0.085)
- Model mampu membedakan trajektori positif dan negatif (92% vs 85.3%)


## Keterbatasan Reproduksi

1.  **Pipeline terfragmentasi.** Tidak ada satu repo yang mencakup seluruh
    pipeline dari MIMIC-III mentah ke format PosNegDM. Reproduce ini harus
    mengintegrasikan 3 repo berbeda (Komorowski, Raghu, PosNegDM) dengan
    environment berbeda (Python 2.7, MATLAB, Python 3).
2.  **Preprocessing tidak identik.** Paper menggunakan pipeline preprocessing
    sendiri yang detailnya tidak dipublikasikan. Reproduce ini menggunakan
    pipeline Komorowski + Raghu sebagai pendekatan terdekat.
3.  **GPU terbatas.** RTX 5060 dengan 8GB VRAM membutuhkan penurunan batch
    size test dari 563 ke 128. Paper kemungkinan menggunakan GPU dengan VRAM
    lebih besar.
4.  **Pretrained mortality classifier.** Model `NNfinal.pth` dari repo
    PosNegDM digunakan apa adanya tanpa dilatih ulang dengan data MIMIC-III
    v1.4.

## Struktur Repository

```
.
├── train.py                       # Main entry point training
├── utils.py                       # Helper utilities (read_dst, dll)
├── nn_train.py                    # Standalone mortality classifier training
├── requirements.txt               # Dependencies
├── environment.yml                # Frozen conda environment (asli)
├── README.MD                      # Dokumentasi ini
├── .gitignore
├── data/
│   ├── preprocessing/             # Script pipeline (notebook + bridge)
│   │   ├── AIClinician_Data_extract_MIMIC3_140219.ipynb
│   │   ├── AIClinician_sepsis3_def_160219.m
│   │   ├── AIClinician_mimic3_dataset_160219.m
│   │   ├── process_interventions.ipynb
│   │   ├── preprocess_data.ipynb
│   │   └── bridge_to_posnegdm.py
│   ├── extracted/                 # Output SQL extraction (ignored)
│   ├── cohort/                    # Output MATLAB cohort (ignored)
│   ├── raw/                       # Output MATLAB dataset (ignored)
│   ├── processed/                 # Output Raghu preprocessing (ignored)
│   ├── final/                     # Output bridge (ignored)
│   └── NNfinal.pth                # Pretrained mortality classifier
├── dualsight/                     # Decision Transformer stack
│   ├── models/
│   ├── training/
│   └── evaluation/
├── nnclsfier/                     # Mortality classifier
└── assets/                        # Gambar untuk README
```

## Cara Menjalankan

```bash
# Setup environment (Python 3.10+ dengan PyTorch)
pip install -r requirements.txt

# Bridge data (jika belum ada di data/final/)
python data/preprocessing/bridge_to_posnegdm.py

# Training
python train.py --dst sepsis_final_data_withTimes.csv --batch_size 32
```

Parameter tambahan:
```
--pos_only           Hanya gunakan trajektori positif (default: False)
--mode               'normal' atau 'delayed' (default: delayed)
--K                  Context window (default: 10)
--batch_size         Training batch size (default: 64, turunkan jika OOM)
--model_type         'dt' (Decision Transformer) atau 'bc' (default: dt)
--embed_dim          Hidden dimension (default: 128)
--n_layer            Jumlah transformer layers (default: 3)
--n_head             Jumlah attention heads (default: 1)
--learning_rate      Learning rate (default: 1e-4)
--max_iters          Jumlah iterasi training (default: 20)
--num_steps_per_iter Steps per iterasi (default: 10000)
```

## Referensi

1. Tamboli, D., Chen, J., Jotheeswaran, K. P., Yu, D., & Aggarwal, V.
   (2024). Reinforced Sequential Decision-Making for Sepsis Treatment: The
   PosNegDM Framework with Mortality Classifier and Transformer. IEEE JBHI.
   https://arxiv.org/abs/2403.07309
2. Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C., & Faisal, A. A.
   (2018). The Artificial Intelligence Clinician learns optimal treatment
   strategies for sepsis in intensive care. Nature Medicine, 24(11),
   1716-1720.
3. Raghu, A., Komorowski, M., Ahmed, I., Celi, L. A., Szolovits, P., &
   Ghassemi, M. (2017). Deep Reinforcement Learning for Sepsis Treatment.
   NeurIPS 2017.
