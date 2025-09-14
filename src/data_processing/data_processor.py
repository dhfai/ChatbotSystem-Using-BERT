import pandas as pd
import json
import os
from typing import List, Dict
from src.config import Config

class DataProcessor:
    """
    Processor untuk mengolah data JSON fakultas menjadi dokumen untuk indexing
    """

    def __init__(self):
        self.documents = []
        self.metadata = []

    def process_fakultas_json_data(self, data_folder: str = None) -> List[str]:
        """
        Memproses data fakultas dari file JSON baru
        """
        # Set default data folder dengan path absolut
        if data_folder is None:
            # Get current working directory dan cari folder data
            current_dir = os.getcwd()
            # Check if we're in pengujian folder
            if current_dir.endswith('pengujian'):
                data_folder = os.path.join(os.path.dirname(current_dir), 'data')
            else:
                data_folder = os.path.join(current_dir, 'data')

        json_files = ['data_feb.json', 'data_fkip.json', 'data_ft.json']
        documents = []

        print(f"Looking for JSON files in: {data_folder}")

        for json_file in json_files:
            file_path = os.path.join(data_folder, json_file)
            print(f"Checking file: {file_path}")
            if os.path.exists(file_path):
                print(f"Processing file: {json_file}")
                documents_from_file = self._process_single_fakultas_json(file_path)
                documents.extend(documents_from_file)
            else:
                print(f"File not found: {file_path}")

        self.documents.extend(documents)
        print(f"Processed {len(documents)} documents from JSON files")
        return documents

    def _process_single_fakultas_json(self, file_path: str) -> List[str]:
        """
        Memproses satu file JSON fakultas
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            fakultas_name = data.get('fakultas', '')

            # 1. Informasi Dasar Fakultas
            if fakultas_name:
                basic_info = f"Fakultas {fakultas_name}"

                # Tambahkan sejarah
                if 'sejarah' in data and data['sejarah']:
                    basic_info += f". Sejarah: {data['sejarah']}"

                # Tambahkan visi
                if 'visi' in data and data['visi']:
                    basic_info += f". Visi: {data['visi']}"

                # Tambahkan misi
                if 'misi' in data and data['misi']:
                    if isinstance(data['misi'], list):
                        misi_text = "; ".join(data['misi'])
                        basic_info += f". Misi: {misi_text}"

                documents.append(basic_info)
                self._add_metadata("basic_info", fakultas_name, file_path, len(documents) - 1)

            # 2. Informasi Pimpinan
            if 'pimpinan' in data and data['pimpinan']:
                pimpinan_doc = f"Pimpinan {fakultas_name}:"

                if 'dekan' in data['pimpinan'] and data['pimpinan']['dekan']:
                    pimpinan_doc += f" Dekan: {data['pimpinan']['dekan']}"

                if 'wakil_dekan' in data['pimpinan'] and data['pimpinan']['wakil_dekan']:
                    wakil_dekan = data['pimpinan']['wakil_dekan']
                    for posisi, nama in wakil_dekan.items():
                        if nama:
                            pimpinan_doc += f". Wakil Dekan {posisi}: {nama}"

                documents.append(pimpinan_doc)
                self._add_metadata("leadership", fakultas_name, file_path, len(documents) - 1)

            # 3. Program Studi
            if 'program_studi' in data and data['program_studi']:
                for prodi_name, prodi_info in data['program_studi'].items():
                    prodi_doc = f"Program Studi {prodi_name} di {fakultas_name}"

                    # Informasi program studi
                    if isinstance(prodi_info, dict) and 'informasi' in prodi_info:
                        prodi_doc += f". {prodi_info['informasi']}"

                    # Akreditasi
                    if isinstance(prodi_info, dict) and 'akreditasi' in prodi_info:
                        akred = prodi_info['akreditasi']
                        if isinstance(akred, dict) and 'akreditasi' in akred:
                            prodi_doc += f" Akreditasi: {akred['akreditasi']}"
                            if 'sk' in akred and akred['sk']:
                                prodi_doc += f", SK: {akred['sk']}"

                    # Fasilitas
                    if isinstance(prodi_info, dict) and 'fasilitas' in prodi_info:
                        fasilitas = prodi_info['fasilitas']
                        if isinstance(fasilitas, list) and fasilitas:
                            clean_fasilitas = [f for f in fasilitas if f and f.strip()]
                            if clean_fasilitas:
                                prodi_doc += f" Fasilitas: {'; '.join(clean_fasilitas)}"

                    # Visi Misi Program Studi
                    if isinstance(prodi_info, dict) and 'visi_misi' in prodi_info:
                        visi_misi = prodi_info['visi_misi']
                        if isinstance(visi_misi, dict):
                            if 'visi' in visi_misi and visi_misi['visi']:
                                prodi_doc += f" Visi: {visi_misi['visi']}"
                            if 'misi' in visi_misi and visi_misi['misi']:
                                if isinstance(visi_misi['misi'], list):
                                    misi_prodi = "; ".join(visi_misi['misi'])
                                    prodi_doc += f" Misi: {misi_prodi}"

                    documents.append(prodi_doc)
                    self._add_metadata("program_studi", fakultas_name, file_path, len(documents) - 1, prodi_name)

            # 4. Organisasi Mahasiswa
            if 'organisasi_mahasiswa' in data and data['organisasi_mahasiswa']:
                org_data = data['organisasi_mahasiswa']
                if 'hmj' in org_data and org_data['hmj']:
                    org_doc = f"Organisasi mahasiswa di {fakultas_name}: "
                    if isinstance(org_data['hmj'], list):
                        org_doc += ", ".join(org_data['hmj'])

                    if 'fungsi' in org_data and org_data['fungsi']:
                        org_doc += f". Fungsi: {org_data['fungsi']}"

                    documents.append(org_doc)
                    self._add_metadata("organisasi", fakultas_name, file_path, len(documents) - 1)

            # 5. Kerjasama Internasional
            if 'kerjasama_internasional' in data and data['kerjasama_internasional']:
                kerjasama = data['kerjasama_internasional']
                if kerjasama and any(kerjasama.values()):  # Ada data kerjasama
                    kerjasama_doc = f"Kerjasama internasional {fakultas_name}:"

                    if 'mitra' in kerjasama and kerjasama['mitra']:
                        if isinstance(kerjasama['mitra'], list):
                            kerjasama_doc += f" Mitra: {', '.join(kerjasama['mitra'])}"

                    if 'kegiatan' in kerjasama and kerjasama['kegiatan']:
                        kerjasama_doc += f". Kegiatan: {kerjasama['kegiatan']}"

                    if 'program' in kerjasama and kerjasama['program']:
                        if isinstance(kerjasama['program'], list) and kerjasama['program']:
                            kerjasama_doc += f". Program: {', '.join(kerjasama['program'])}"

                    documents.append(kerjasama_doc)
                    self._add_metadata("kerjasama", fakultas_name, file_path, len(documents) - 1)

            # 6. Pendidikan Berbasis Islam
            if 'pendidikan_berbasis_islam' in data and data['pendidikan_berbasis_islam']:
                islam_doc = f"Pendidikan berbasis Islam di {fakultas_name}: {data['pendidikan_berbasis_islam']}"
                documents.append(islam_doc)
                self._add_metadata("pendidikan_islam", fakultas_name, file_path, len(documents) - 1)

            return documents

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _add_metadata(self, doc_type: str, fakultas: str, source_file: str, doc_index: int, extra_info: str = ""):
        """
        Menambahkan metadata untuk dokumen
        """
        metadata = {
            "source": os.path.basename(source_file),
            "type": doc_type,
            "fakultas": fakultas,
            "doc_index": doc_index
        }
        if extra_info:
            metadata["extra_info"] = extra_info

        self.metadata.append(metadata)

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
