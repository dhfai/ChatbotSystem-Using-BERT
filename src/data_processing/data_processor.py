import pandas as pd
import json
from typing import List, Dict
from src.config import Config

class DataProcessor:
    """
    Processor untuk mengolah data CSV dan Excel menjadi dokumen untuk indexing
    """

    def __init__(self):
        self.documents = []
        self.metadata = []

    def process_biaya_kuliah_data(self, file_path: str = None) -> List[str]:
        """
        Memproses data biaya kuliah dari CSV dengan format:
        NO,FAKULTAS DAN PROGRAM STUDI,AKREDITASI,UANG KULIAH,UANG PEMBANGUNAN
        HANYA menggunakan data mentah tanpa modifikasi
        """
        file_path = file_path or Config.DATA_BIAYA_PATH

        try:
            df = pd.read_csv(file_path)
            documents = []

            for idx, row in df.iterrows():
                # Buat dokumen text yang informatif dari data biaya kuliah TANPA modifikasi
                program_studi = str(row['FAKULTAS DAN PROGRAM STUDI']).strip()
                akreditasi = str(row['AKREDITASI']).strip()
                uang_kuliah = str(row['UANG KULIAH']).strip()
                uang_pembangunan = str(row['UANG PEMBANGUNAN']).strip()

                # Buat dokumen dengan format standar TANPA menambah kata kunci multimedia
                doc_text = f"Program Studi {program_studi}"

                if akreditasi and akreditasi != 'nan' and akreditasi != '':
                    doc_text += f" dengan akreditasi {akreditasi}"
                if uang_kuliah and uang_kuliah != 'nan' and uang_kuliah != '–' and uang_kuliah != '':
                    doc_text += f" memiliki uang kuliah per semester {uang_kuliah}"
                if uang_pembangunan and uang_pembangunan != 'nan' and uang_pembangunan != '–' and uang_pembangunan != '':
                    doc_text += f" dan uang pembangunan {uang_pembangunan}"

                documents.append(doc_text)

                # Tambahkan metadata
                metadata = {
                    "source": "biaya_kuliah",
                    "row_id": idx + 1,
                    "type": "financial_info",
                    "program_studi": program_studi,
                    "akreditasi": akreditasi,
                    "uang_kuliah": uang_kuliah,
                    "uang_pembangunan": uang_pembangunan
                }
                self.metadata.append(metadata)

            self.documents.extend(documents)
            print(f"Processed {len(documents)} biaya kuliah documents")
            return documents

        except Exception as e:
            print(f"Error processing biaya kuliah data: {e}")
            return []

    def process_fakultas_data(self, file_path: str = None) -> List[str]:
        """
        Memproses data fakultas dari CSV yang sudah diperbaiki (datafux_fixed.csv)
        Menggantikan proses Excel dengan CSV yang lebih konsisten
        """
        # Gunakan file CSV yang sudah diperbaiki sebagai default
        if file_path is None:
            file_path = "data/datafux_fixed.csv"

        try:
            df = pd.read_csv(file_path)
            documents = []

            for idx, row in df.iterrows():
                fakultas_id = str(row['id']).strip()
                fakultas_name = str(row['fakultas']).strip()
                informasi = str(row['informasi']).strip()

                # Format dokumen dengan struktur yang jelas
                doc_text = f"{fakultas_name}: {informasi}"

                if doc_text and doc_text.strip():
                    documents.append(doc_text)

                    metadata = {
                        "source": "fakultas_csv",
                        "row_id": idx,
                        "type": "academic_info",
                        "fakultas_id": fakultas_id,
                        "fakultas_name": fakultas_name
                    }
                    self.metadata.append(metadata)

            self.documents.extend(documents)
            print(f"Processed {len(documents)} fakultas documents from CSV")
            return documents

        except Exception as e:
            print(f"Error processing fakultas CSV data: {e}")
            return self._extract_fakultas_from_biaya_data()

    def _extract_fakultas_from_biaya_data(self) -> List[str]:
        """
        Extract informasi fakultas dari data program studi di biaya kuliah
        """
        try:
            df = pd.read_csv(Config.DATA_BIAYA_PATH)
            fakultas_info = {}
            documents = []

            for idx, row in df.iterrows():
                program_studi = str(row['FAKULTAS DAN PROGRAM STUDI']).strip()

                if 'Pendidikan' in program_studi and 'Dokter' not in program_studi:
                    fakultas = "Fakultas Keguruan dan Ilmu Pendidikan"
                elif 'Dokter' in program_studi or 'Farmasi' in program_studi or 'Kebidanan' in program_studi or 'Keperawatan' in program_studi:
                    fakultas = "Fakultas Kedokteran dan Ilmu Kesehatan"
                elif 'Ekonomi' in program_studi or 'Manajemen' in program_studi or 'Akuntansi' in program_studi:
                    fakultas = "Fakultas Ekonomi dan Bisnis"
                elif 'Teknik' in program_studi or 'Informatika' in program_studi or 'Arsitektur' in program_studi:
                    fakultas = "Fakultas Teknik"
                elif 'Agama Islam' in program_studi or 'Syariah' in program_studi:
                    fakultas = "Fakultas Agama Islam"
                elif 'Hukum' in program_studi:
                    fakultas = "Fakultas Hukum"
                else:
                    fakultas = "Program Studi Lainnya"

                if fakultas not in fakultas_info:
                    fakultas_info[fakultas] = []
                fakultas_info[fakultas].append(program_studi)

            for fakultas, programs in fakultas_info.items():
                doc_text = f"{fakultas} memiliki program studi: {', '.join(programs[:5])}"
                if len(programs) > 5:
                    doc_text += f" dan {len(programs) - 5} program studi lainnya"

                documents.append(doc_text)

                metadata = {
                    "source": "fakultas_extracted",
                    "type": "academic_info",
                    "fakultas": fakultas,
                    "program_count": len(programs)
                }
                self.metadata.append(metadata)

            self.documents.extend(documents)
            print(f"Extracted {len(documents)} fakultas documents from biaya kuliah data")
            return documents

        except Exception as e:
            print(f"Error extracting fakultas data: {e}")
            return []

    def add_custom_documents(self, custom_docs: List[str], doc_type: str = "custom"):
        """
        Menambahkan dokumen custom
        """
        for idx, doc in enumerate(custom_docs):
            self.documents.append(doc)
            metadata = {
                "source": "custom",
                "doc_id": idx,
                "type": doc_type
            }
            self.metadata.append(metadata)

    def get_all_documents(self) -> tuple:
        """
        Mendapatkan semua dokumen dan metadata
        """
        return self.documents, self.metadata

    def save_processed_data(self, output_path: str = "data/processed_documents.json"):
        """
        Menyimpan dokumen yang sudah diproses
        """
        data = {
            "documents": self.documents,
            "metadata": self.metadata
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.documents)} documents to {output_path}")

    def load_processed_data(self, input_path: str = "data/processed_documents.json"):
        """
        Memuat dokumen yang sudah diproses
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.documents = data["documents"]
            self.metadata = data["metadata"]

            print(f"Loaded {len(self.documents)} documents from {input_path}")
            return True

        except Exception as e:
            print(f"Error loading processed data: {e}")
            return False
