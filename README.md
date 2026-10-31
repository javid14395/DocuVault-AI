# DocuVault-AI
A 100% offline, GPU-accelerated desktop AI vault for targeted facial recognition and OCR-based ID extraction.
# 🗄️ DocuVault AI

> A 100% offline, GPU-accelerated desktop AI vault for targeted facial recognition and OCR-based ID extraction.

DocuVault AI is a powerful, privacy-first desktop application designed to organize, scan, and extract data from local documents and images. It runs entirely on your local machine, ensuring **zero cloud uploads** and complete data privacy. Built with a high-performance Python backend and a sleek, native desktop UI, it seamlessly handles thousands of images.

## ✨ Key Features

* **Offline AI Engine:** Powered by local `InsightFace` (buffalo_l/buffalo_s) and `RapidOCR`. No internet connection required.
* **CUDA GPU Acceleration:** Fully utilizes NVIDIA GPUs (RTX series) for lightning-fast facial embeddings and OCR processing.
* **Targeted Face Search:** Crop a face from a group photo and instruct the AI to find matches across your entire vault or within a specific local folder.
* **ID Vault Mode:** Automatically identifies, extracts, and isolates official documents (Aadhaar, PAN, Passports, DL) while hiding normal gallery photos.
* **Square Face Cropper UI:** WhatsApp/Instagram style manual cropping tool for pinpoint accuracy before searching.
* **Smart Clustering:** Automatically groups photos of the same person together based on cosine similarity thresholds.

## 🛠️ Tech Stack

* **Backend:** Python 3, FastAPI, SQLite (Encrypted/SQLCipher ready)
* **Desktop Window:** `pywebview` (Native Windows/macOS rendering)
* **AI & Vision:** InsightFace, RapidOCR, OpenCV, NumPy
* **Frontend:** HTML, JavaScript, Tailwind CSS (Offline Assets), Cropper.js

## 🚀 Future Roadmap (Help Wanted!)

We are looking for open-source contributors to help build the following features:
- [ ] Virtual Scrolling / Pagination for massive image grids to prevent DOM overload.
- [ ] Junk Face Filter (ignoring faces smaller than 80x80 pixels).
- [ ] EXIF Metadata extraction (Sort photos by original Date Taken).
- [ ] Background Async Processing with WebSockets for real-time progress bars.

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
