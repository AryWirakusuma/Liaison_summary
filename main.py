import streamlit as st
import pandas as pd
import os
from langchain_community.llms import Ollama
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Database untuk memetakan nama perusahaan ke sektor atau jenis usaha (LU)
company_lu_mapping = {
    'Subak Uma Dalem': 'Pertanian, Kehutanan dan Perikanan',
    'Subak Keraman': 'Pertanian, Kehutanan dan Perikanan',
    'Limajari Interbhuana': 'Transportasi dan Pergudangan',
    'Putra Bhineka Perkasa, PT': 'Industri Pengolahan',
    'Sumiati Ekspor Internasional': 'Perdagangan Besar dan Eceran',
    'Hotel Merusaka Nusa Dua': 'Akomodasi dan Makan Minum',
    'Lina Jaya, CV': 'Konstruksi',
    'Anugerah Merta Sedana, PT': 'Industri Pengolahan',
    'Jasamarga Bali Tol, PT': 'Transportasi dan Pergudangan',
    'Lotte Grosir Bali': 'Perdagangan Besar dan Eceran',
    'The Mulia Hotels and Resorts': 'Akomodasi dan Makan Minum',
    'The Laguna Resort': 'Akomodasi dan Makan Minum',
    'Hotel Grand Hyatt': 'Akomodasi dan Makan Minum',
    'Bank Pembangunan Daerah Bali': 'Jasa Keuangan',
    'Lion Mentari Airlines KC': 'Transportasi dan Pergudangan',
    'Sheraton Bali Kuta Resort': 'Akomodasi dan Makan Minum',
    'Sea Six Energy Indonesia, PT': 'Industri Pengolahan',
    'Subak Aseman IV': 'Pertanian, Kehutanan dan Perikanan'
}


def preprocess_text(text, remove_stopwords=True):
    # Hapus links
    text = re.sub(r"http\S+", "", str(text))
    # Hapus nomor dan karakter khusus
    text = re.sub("[^A-Za-z]+", " ", str(text))
    # Hapus stopwords
    if remove_stopwords:
        # 1. Membuat Token
        tokens = nltk.word_tokenize(text)
        # 2. Mengecek token apabila terdapat stopwords maka dihapus
        stopwords_ind = set(nltk.corpus.stopwords.words("indonesian"))
        stopwords_ind.update(['com', 'rp', 'm', 'ol', 'triwulan', 'perusahaan'])  # Tambahan stopwords
        tokens = [w for w in tokens if not w.lower() in stopwords_ind]
        # 3. Mengabungkan token kembali
        text = " ".join(tokens)
    # mereturn text yang sudah dicleaning dengan huruf kecil
    text = text.lower().strip()
    return text


def generate_wordcloud(text, title):
    wordcloud = WordCloud(stopwords=nltk.corpus.stopwords.words("indonesian"),
                          background_color='white',
                          width=1000, height=1000
                          ).generate(text)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)


def calculate_tfidf(corpus):
    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words("indonesian"))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


def display_top_tfidf_words(tfidf_matrix, feature_names, top_n=10):
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(feature_names):
        data.append((term, sums[0, col]))
    ranking = sorted(data, key=lambda x: x[1], reverse=True)
    top_words = ranking[:top_n]
    st.write(f"Top {top_n} words by TF-IDF:")
    for word, score in top_words:
        st.write(f"{word}: {score:.4f}")


def process_excel_file(uploaded_file, is_second_excel=False):
    combined_df = pd.DataFrame(columns=['Nama Contact', 'Pertanyaan', 'Nilai'])

    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        st.header(f'Data pada Sheet: {sheet_name}')
        st.write("Dataframe utuh sebelum penyaringan:")
        st.dataframe(df)  # Tampilkan dataframe utuh sebelum penyaringan

        try:
            filtered_df = df[['Unnamed: 2', 'Unnamed: 3']]  # Ambil kolom Unnamed: 2 dan Unnamed: 3
            filtered_df.columns = ['Pertanyaan', 'Nilai']  # Ganti nama kolom agar sesuai

            # Filter hanya baris yang mengandung pertanyaan yang diinginkan
            relevant_columns = ['Permintaan Domestik  - Likert Scale',
                                'Permintaan Ekspor  - Likert Scale',
                                'Kapasitas Utilisasi - Likert Scale',
                                'Persediaan - Likert Scale',
                                'Investasi - Likert Scale',
                                'Biaya Energi - Likert Scale',
                                'Biaya Tenaga Kerja (Upah) - Likert Scale',
                                'Harga Jual – Likert Scale',
                                'Margin Usaha - Likert Scale',
                                'Tenaga Kerja - Likert Scale',
                                'Perkiraan Penjualan – Likert Scale',
                                'Perkiraan Tingkat Upah – Likert Scale',
                                'Perkiraan Harga Jual – Likert Scale',
                                'Perkiraan Jumlah Tenaga Kerja – Likert Scale',
                                'Perkiraan Investasi – Likert Scale']

            filtered_df = filtered_df[filtered_df['Pertanyaan'].isin(relevant_columns)]

            # Ubah tanda minus (–) menjadi strip (-) pada kolom Pertanyaan
            filtered_df['Pertanyaan'] = filtered_df['Pertanyaan'].str.replace('–', '-')

            # Tambahkan kolom 'Nama Contact' dengan nama sheet
            filtered_df.insert(0, 'Nama Contact', sheet_name)

            # Mencocokkan nama perusahaan dengan LU dari database
            filtered_df['LU'] = filtered_df['Nama Contact'].apply(
                lambda x: next((lu for company, lu in company_lu_mapping.items() if company in x), 'Unknown'))

            # Gabungkan dataframe hasil filter ke dalam dataframe kombinasi
            combined_df = pd.concat([combined_df, filtered_df])

            # Mengganti nilai kosong dengan kata "kosong" di combined_df
            combined_df = combined_df.fillna('kosong')
        except KeyError:
            st.error("Kolom yang dibutuhkan tidak ditemukan dalam file Excel.")

    if not combined_df.empty:
        st.header('Dataframe setelah penyaringan dari semua sheet:')
        st.dataframe(combined_df)

        st.header('Rata-rata nilai untuk setiap kolom Likert Scale:')
        avg_values = {}
        # Iterasi melalui setiap kolom Likert Scale
        for col in combined_df['Pertanyaan'].unique():
            # Konversi nilai-nilai ke tipe data numerik
            numeric_values = pd.to_numeric(combined_df[combined_df['Pertanyaan'] == col]['Nilai'], errors='coerce')
            # Hitung rata-rata nilai untuk kolom tersebut
            avg_value = numeric_values.mean()
            avg_values[col] = avg_value if not pd.isna(avg_value) else None

        # Tampilkan rata-rata nilai untuk setiap kolom Likert Scale
        for col, avg_value in avg_values.items():
            # Ubah "- Likert Scale" menjadi " "
            col_name = col.replace(' - Likert Scale', ' ')
            st.write(f"{col_name}: {avg_value}")

        if is_second_excel:
            # Set up dictionary to store total LU for each sector
            lu_totals = {}

            # Iterate through each company in the uploaded Excel file
            for company in combined_df['Nama Contact'].unique():
                # Get LU for the current company from the mapping
                lu = company_lu_mapping.get(company, 'Unknown')

                # Add LU to total count for the corresponding sector
                if lu in lu_totals:
                    lu_totals[lu] += 1
                else:
                    lu_totals[lu] = 1

            # Print total LU for each sector
            st.header('Total LU untuk setiap sektor atau jenis usaha (LU):')
            for lu, total in lu_totals.items():
                st.write(f"{lu}: {total}")

            # Calculate LU dominance
            total_lu = sum(lu_totals.values())
            lu_dominance = {lu: (count / total_lu) * 100 for lu, count in lu_totals.items()}
            max_dominant_lu = max(lu_dominance, key=lu_dominance.get)

            st.header('LU yang Mendominasi:')
            st.write(f"{max_dominant_lu}: {lu_dominance[max_dominant_lu]:.2f}%")

    return combined_df, avg_values


def process_domestik_ekspor_df(uploaded_file):
    combined_df_domestik_ekspor = pd.DataFrame(columns=['Nama Contact', 'Pertanyaan', 'Nilai', 'LU'])
    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        try:
            relevant_columns = ['Permintaan Domestik  - Likert Scale', 'Permintaan Ekspor  - Likert Scale']
            df_filtered = df[['Unnamed: 2', 'Unnamed: 3']]  # Ambil kolom Unnamed: 2 dan Unnamed: 3
            df_filtered.columns = ['Pertanyaan', 'Nilai']  # Ganti nama kolom agar sesuai
            df_filtered = df_filtered[df_filtered['Pertanyaan'].isin(relevant_columns)]

            # Tambahkan kolom 'Nama Contact' dengan nama sheet
            df_filtered.insert(0, 'Nama Contact', sheet_name)

            # Mencocokkan nama perusahaan dengan LU dari database
            df_filtered['LU'] = df_filtered['Nama Contact'].apply(
                lambda x: next((lu for company, lu in company_lu_mapping.items() if company in x), 'Unknown'))

            # Gabungkan dataframe hasil filter ke dalam dataframe kombinasi
            combined_df_domestik_ekspor = pd.concat([combined_df_domestik_ekspor, df_filtered])

            # Mengganti nilai kosong dengan kata "kosong"
            combined_df_domestik_ekspor = combined_df_domestik_ekspor.fillna('kosong')
        except KeyError:
            st.error("Kolom yang dibutuhkan tidak ditemukan dalam file Excel.")

    # Menambahkan perhitungan jumlah orientasi domestik, ekspor, dan domestik & ekspor
    domestic_count = 0
    export_count = 0
    domestic_export_count = 0

    lu_domestic = []
    lu_export = []
    lu_domestic_export = []

    for contact in combined_df_domestik_ekspor['Nama Contact'].unique():
        df_contact = combined_df_domestik_ekspor[combined_df_domestik_ekspor['Nama Contact'] == contact]
        domestic_value = df_contact[df_contact['Pertanyaan'] == 'Permintaan Domestik  - Likert Scale']['Nilai'].values
        export_value = df_contact[df_contact['Pertanyaan'] == 'Permintaan Ekspor  - Likert Scale']['Nilai'].values

        lu_contact = df_contact['LU'].values[0]

        if len(domestic_value) > 0 and len(export_value) > 0:
            if domestic_value[0] != 'kosong' and export_value[0] == 'kosong':
                domestic_count += 1
                lu_domestic.append(lu_contact)
            elif domestic_value[0] == 'kosong' and export_value[0] != 'kosong':
                export_count += 1
                lu_export.append(lu_contact)
            elif domestic_value[0] != 'kosong' and export_value[0] != 'kosong':
                domestic_export_count += 1
                lu_domestic_export.append(lu_contact)

    total_count = domestic_count + export_count + domestic_export_count

    if total_count > 0:
        domestic_percentage = (domestic_count / total_count) * 100
        export_percentage = (export_count / total_count) * 100
        domestic_export_percentage = (domestic_export_count / total_count) * 100
    else:
        domestic_percentage = export_percentage = domestic_export_percentage = 0

    st.write(f"Jumlah orientasi Domestik: {domestic_count}")
    st.write(f"Jumlah orientasi Ekspor: {export_count}")
    st.write(f"Jumlah orientasi Domestik dan Ekspor: {domestic_export_count}")

    st.write(f"Persen orientasi Domestik: {domestic_percentage:.2f}%")
    st.write(f"Persen orientasi Ekspor: {export_percentage:.2f}%")
    st.write(f"Persen orientasi Domestik dan Ekspor: {domestic_export_percentage:.2f}%")

    lu_domestic = set(lu_domestic)
    lu_export = set(lu_export)
    lu_domestic_export = set(lu_domestic_export)

    st.write(f"LU yang berorientasi Domestik: {', '.join(lu_domestic)}")
    st.write(f"LU yang berorientasi Ekspor: {', '.join(lu_export)}")
    st.write(f"LU yang berorientasi Domestik dan Ekspor: {', '.join(lu_domestic_export)}")

    return combined_df_domestik_ekspor, total_count, domestic_count, export_count, domestic_export_count, lu_domestic, lu_export, lu_domestic_export


def process_alasan_domestik_ekspor_df(uploaded_file):
    alasan_domestik_ekspor_df = pd.DataFrame(columns=['Nama Contact', 'Permintaan Domestik', 'Permintaan Ekspor'])
    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        try:
            # Ambil kolom Permintaan Domestik dan Permintaan Ekspor
            permintaan_domestik = \
            df[df['Unnamed: 2'] == 'Permintaan/Penjualan  - Permintaan Domestik']['Unnamed: 3'].values[0]
            permintaan_ekspor = df[df['Unnamed: 2'] == 'Permintaan/Penjualan - Permintaan Ekspor']['Unnamed: 3'].values[
                0]

            # Tambahkan data ke dalam dataframe alasan_domestik_ekspor_df
            alasan_domestik_ekspor_df = pd.concat([alasan_domestik_ekspor_df, pd.DataFrame({
                'Nama Contact': [sheet_name],
                'Permintaan Domestik': [permintaan_domestik],
                'Permintaan Ekspor': [permintaan_ekspor]
            })], ignore_index=True)
        except (KeyError, IndexError):
            st.error(f"Data Permintaan Domestik atau Ekspor tidak ditemukan dalam sheet {sheet_name}")

    st.header('Dataframe Alasan Domestik dan Ekspor:')
    st.dataframe(alasan_domestik_ekspor_df)

    # Menjabarkan jawaban pada kolom Permintaan Domestik dan Permintaan Ekspor
    jawaban_domestik = " ".join([f"Pada triwulan laporan, kontak menyatakan bahwa {jawaban}." for jawaban in
                                 alasan_domestik_ekspor_df['Permintaan Domestik']])
    jawaban_ekspor = " ".join([f"Pada triwulan laporan, kontak menyatakan bahwa {jawaban}." for jawaban in
                               alasan_domestik_ekspor_df['Permintaan Ekspor']])

    st.header('Jawaban Dijabarkan:')
    st.subheader('Penjabaran Permintaan Domestik:')
    st.write(jawaban_domestik)
    st.subheader('Penjabaran Permintaan Ekspor:')
    st.write(jawaban_ekspor)

    return alasan_domestik_ekspor_df, jawaban_domestik, jawaban_ekspor


def process_additional_data_df(uploaded_file):
    additional_data_df = pd.DataFrame(columns=['Nama Contact', 'Kapasitas Utilisasi', 'Persediaan', 'Investasi',
                                               'Biaya-biaya - Bahan Baku Diluar Gaji/Upah', 'Biaya Energi',
                                               'Biaya Tenaga Kerja (Upah)', 'Harga Jual - Perkembangan Harga Jual',
                                               'Margin Usaha', 'Perkembangan Jumlah Tenaga Kerja',
                                               'Perkiraan Penjualan', 'Perkiraan Tingkat Upah', 'Perkiraan Harga Jual',
                                               'Perkiraan Jumlah Tenaga Kerja', 'Perkiraan Investasi',
                                               'Pembiayaan dan Suku Bunga'])

    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        try:
            data = {
                'Nama Contact': sheet_name,
                'Kapasitas Utilisasi': df[df['Unnamed: 2'] == 'Kapasitas Utilisasi']['Unnamed: 3'].values[0],
                'Persediaan': df[df['Unnamed: 2'] == 'Persediaan']['Unnamed: 3'].values[0],
                'Investasi': df[df['Unnamed: 2'] == 'Investasi']['Unnamed: 3'].values[0],
                'Biaya-biaya - Bahan Baku Diluar Gaji/Upah':
                    df[df['Unnamed: 2'] == 'Biaya-biaya - Bahan Baku Diluar Gaji/Upah']['Unnamed: 3'].values[0],
                'Biaya Energi': df[df['Unnamed: 2'] == 'Biaya Energi']['Unnamed: 3'].values[0],
                'Biaya Tenaga Kerja (Upah)': df[df['Unnamed: 2'] == 'Biaya Tenaga Kerja (Upah)']['Unnamed: 3'].values[
                    0],
                'Harga Jual - Perkembangan Harga Jual':
                    df[df['Unnamed: 2'] == 'Harga Jual - Perkembangan Harga Jual']['Unnamed: 3'].values[0],
                'Margin Usaha': df[df['Unnamed: 2'] == 'Margin Usaha']['Unnamed: 3'].values[0],
                'Perkembangan Jumlah Tenaga Kerja':
                    df[df['Unnamed: 2'] == 'Perkembangan Jumlah Tenaga Kerja']['Unnamed: 3'].values[0],
                'Perkiraan Penjualan': df[df['Unnamed: 2'] == 'Perkiraan Penjualan']['Unnamed: 3'].values[0],
                'Perkiraan Tingkat Upah': df[df['Unnamed: 2'] == 'Perkiraan Tingkat Upah']['Unnamed: 3'].values[0],
                'Perkiraan Harga Jual': df[df['Unnamed: 2'] == 'Perkiraan Harga Jual']['Unnamed: 3'].values[0],
                'Perkiraan Jumlah Tenaga Kerja':
                    df[df['Unnamed: 2'] == 'Perkiraan Jumlah Tenaga Kerja']['Unnamed: 3'].values[0],
                'Perkiraan Investasi': df[df['Unnamed: 2'] == 'Perkiraan Investasi']['Unnamed: 3'].values[0],
                'Pembiayaan dan Suku Bunga': df[df['Unnamed: 2'] == 'Pembiayaan dan Suku Bunga']['Unnamed: 3'].values[
                    0],
            }

            additional_data_df = pd.concat([additional_data_df, pd.DataFrame(data, index=[0])], ignore_index=True)
        except (KeyError, IndexError):
            st.error(f"Data yang dibutuhkan tidak ditemukan dalam sheet {sheet_name}")

    st.header('Dataframe Data Tambahan:')
    st.dataframe(additional_data_df)

    # Menjabarkan jawaban pada kolom-kolom yang relevan
    jawaban_additional_data = {}
    for column in additional_data_df.columns[1:]:
        jawaban_additional_data[column] = " ".join([f"Pada triwulan laporan, kontak menyatakan bahwa {jawaban}."
                                                    for jawaban in additional_data_df[column]])

    st.header('Jawaban Dijabarkan:')
    for column, jawaban in jawaban_additional_data.items():
        st.subheader(f'Penjabaran {column}:')
        st.write(jawaban)

    return additional_data_df, jawaban_additional_data


def get_factors(text, title):
    llm = Ollama(model="openchat")

    prompt = f"{text} tolong cari faktor penurunan {title} lengkap dengan penjelasannya menggunakan bahasa indonesia, ingat menggunakan bahasa indonesia."

    completion = llm(prompt)
    factors = completion

    # Asumsikan ringkasan berisi topik yang dipisahkan oleh koma
    factors_list = factors.split(',')
    top_5_factors = factors_list[:5] if len(factors_list) > 5 else factors_list

    return [factor.strip() for factor in top_5_factors]


def generate_summary(avg_values_1, avg_values_2, jawaban_domestik, jawaban_ekspor, jawaban_additional_data, quarter_now,
                     quarter_before, phenomenon_reason):
    llm = Ollama(model="openchat")

    # Kesimpulan Domestik
    prompt_domestik = (
        f"Berikut adalah data rata-rata nilai antara dua triwulan:\n"
        f"Permintaan Domestik : {avg_values_2.get('Permintaan Domestik  - Likert Scale', 'Tidak ada data')} dibandingkan {avg_values_1.get('Permintaan Domestik  - Likert Scale', 'Tidak ada data')}\n"
        f"Topik Permintaan Domestik:\n{jawaban_domestik}\n"
        f"Silakan buat kesimpulan tentang kinerja domestik pada triwulan {quarter_now} dibandingkan triwulan {quarter_before} berdasarkan data tersebut, tolong menggunakan bahasa indonesia. lalu untuk kata katanya dibuat rapi, sopan, dan netral"
    )

    # Kesimpulan Ekspor
    prompt_ekspor = (
        f"Berikut adalah data rata-rata nilai antara dua triwulan:\n"
        f"Permintaan Ekspor : {avg_values_2.get('Permintaan Ekspor  - Likert Scale', 'Tidak ada data')} dibandingkan {avg_values_1.get('Permintaan Ekspor  - Likert Scale', 'Tidak ada data')}\n"
        f"Topik Permintaan Ekspor:\n{jawaban_ekspor}\n"
        f"Silakan buat kesimpulan tentang kinerja ekspor pada triwulan {quarter_now} dibandingkan triwulan {quarter_before} berdasarkan data tersebut, tolong menggunakan bahasa indonesia. lalu untuk kata katanya dibuat rapi, sopan, dan netral"
    )

    completion_domestik = llm(prompt_domestik)
    completion_ekspor = llm(prompt_ekspor)

    kesimpulan_domestik = completion_domestik
    kesimpulan_ekspor = completion_ekspor

    # Kesimpulan Data Tambahan
    kesimpulan_additional = {}
    for column, jawaban in jawaban_additional_data.items():
        prompt_additional = (
            f"Berikut adalah data rata-rata nilai antara dua triwulan:\n"
            f"{jawaban}\n"
            f"Silakan buat kesimpulan tentang kinerja pada triwulan {quarter_now} dibandingkan triwulan {quarter_before} untuk topik {column} berdasarkan data tersebut, tolong menggunakan bahasa indonesia. lalu untuk kata katanya dibuat rapi, sopan, dan netral"
        )

        completion_additional = llm(prompt_additional)
        kesimpulan_additional[column] = completion_additional

    return kesimpulan_domestik, kesimpulan_ekspor, kesimpulan_additional


def main():
    st.set_page_config(page_title='ML Summary Liaison')
    st.title('Summary of Liaison Report Using AI 📊')

    uploaded_file_1 = st.file_uploader('Upload file Excel Pertama (XLSX) untuk triwulan sebelumnya', type='xlsx')

    if uploaded_file_1:
        st.markdown('---')
        combined_df_1, avg_values_1 = process_excel_file(uploaded_file_1)

        st.write("\\n---\\n")
        st.write("Upload file Excel Kedua (XLSX) untuk triwulan sekarang")
        uploaded_file_2 = st.file_uploader('Upload file Excel Kedua (XLSX)', type='xlsx')

        if uploaded_file_2:
            st.markdown('---')
            combined_df_2, avg_values_2 = process_excel_file(uploaded_file_2, is_second_excel=True)

            # Tambahkan pemrosesan dataframe domestik ekspor
            st.header('Dataframe Permintaan Domestik dan Ekspor:')
            combined_df_domestik_ekspor, total_count, domestic_count, export_count, domestic_export_count, lu_domestic, lu_export, lu_domestic_export = process_domestik_ekspor_df(
                uploaded_file_2)
            st.dataframe(combined_df_domestik_ekspor)

            st.header('')
            alasan_domestik_ekspor_df, jawaban_domestik, jawaban_ekspor = process_alasan_domestik_ekspor_df(
                uploaded_file_2)

            # Tambahkan pemrosesan dataframe data tambahan
            st.header('Dataframe Data Tambahan:')
            additional_data_df, jawaban_additional_data = process_additional_data_df(uploaded_file_2)
            st.dataframe(additional_data_df)

            st.header('Perbandingan Rata-rata nilai antara dua upload:')
            if combined_df_1.empty or combined_df_2.empty:
                st.warning("Salah satu atau kedua dataframe kosong, perbandingan tidak dapat dilakukan.")
            else:
                changes = {'naik': 0, 'turun': 0}
                turun_indicators = []
                change_domestik = 0
                change_ekspor = 0

                # Bandingkan nilai rata-rata antara dua dataframe
                for col in avg_values_1.keys():
                    if col in avg_values_2.keys():
                        change = avg_values_2[col] - avg_values_1[col]
                        if change > 0:
                            st.write(f"{col.replace(' - Likert Scale', ' ')}: Naik sebesar {change}")
                            changes['naik'] += 1
                        elif change < 0:
                            st.write(f"{col.replace(' - Likert Scale', ' ')}: Turun sebesar {abs(change)}")
                            changes['turun'] += 1
                            turun_indicators.append(col.replace(' - Likert Scale', ' '))
                            if col == 'Permintaan Domestik  - Likert Scale':
                                change_domestik = abs(change)
                            if col == 'Permintaan Ekspor  - Likert Scale':
                                change_ekspor = abs(change)
                        else:
                            st.write(f"{col.replace(' - Likert Scale', ' ')}: Tidak mengalami perubahan")

                # Generate comparison sentence
                quarter_now = st.text_input("Triwulan Sekarang (e.g., I 2024, II 2023, III 2022, IV 2024):")
                quarter_before = st.text_input("Triwulan Sebelumnya (e.g., I 2024, II 2023, III 2022, IV 2024):")
                phenomenon_reason = st.text_input(
                    "Alasan Fenomena yang terjadi (e.g., dinamika pasar, kondisi politik, dll):")

                if quarter_now and quarter_before and phenomenon_reason:
                    if changes['naik'] > changes['turun']:
                        trend = "percepatan"
                    elif changes['naik'] < changes['turun']:
                        trend = "perlambatan"
                    else:
                        trend = "tidak ada perubahan"

                    st.header('Kesimpulan pertama:')
                    st.write(
                        f"Kinerja perekonomian Provinsi pada triwulan {quarter_now} terindikasi tumbuh melambat dibandingkan triwulan {quarter_before}. Hal ini sebagaimana tercermin dari hasil likert pada triwulan {quarter_now} yang mengalami {trend} pada {changes['turun']} dari {len(avg_values_1)} indikator likert liaison, yaitu {', '.join(turun_indicators)}. Fenomena ini terjadi karena {phenomenon_reason}.")

                    # Calculate total number of contacts
                    total_contacts = combined_df_domestik_ekspor['Nama Contact'].nunique()

                    # Calculate LU dominance
                    max_dominant_lu = combined_df_domestik_ekspor['LU'].value_counts().idxmax()
                    max_dominant_lu_percentage = (
                            combined_df_domestik_ekspor['LU'].value_counts(normalize=True).max() * 100)

                    # Calculate orientation percentages
                    domestic_percentage = (domestic_count / total_count) * 100 if total_count > 0 else 0
                    export_percentage = (export_count / total_count) * 100 if total_count > 0 else 0
                    domestic_export_percentage = (domestic_export_count / total_count) * 100 if total_count > 0 else 0

                    # Generate second conclusion
                    st.header('Kesimpulan Kedua:')
                    st.write(
                        f"Jumlah total perusahaan yang di liaison KPw Bank Indonesia Provinsi Bali periode triwulan {quarter_now} adalah {total_contacts} kontak. Liaison pada triwulan laporan didominasi oleh LU {max_dominant_lu} sebesar {max_dominant_lu_percentage:.2f}% dari total kontak. Kemudian, {domestic_percentage:.2f}% berorientasi domestik, {export_percentage:.2f}% berorientasi ekspor dan {domestic_export_percentage:.2f}% berorientasi domestik dan ekspor. Perusahaan yang sepenuhnya berorientasi domestik adalah LU {', '.join(lu_domestic)}. Kontak yang sepenuhnya berorientasi domestik dan ekspor adalah LU {', '.join(lu_domestic_export)}. Sedangkan, kontak yang sepenuhnya berorientasi ekspor adalah beberapa kontak pada LU {', '.join(lu_export)}.")

                    st.header('Wordclouds dari Jawaban Dijabarkan:')
                    jawaban_domestik_cleaned = preprocess_text(jawaban_domestik)
                    jawaban_ekspor_cleaned = preprocess_text(jawaban_ekspor)
                    st.subheader('Wordcloud untuk Jawaban Permintaan Domestik:')
                    generate_wordcloud(jawaban_domestik_cleaned, 'Wordcloud untuk Jawaban Permintaan Domestik')
                    st.subheader('Wordcloud untuk Jawaban Permintaan Ekspor:')
                    generate_wordcloud(jawaban_ekspor_cleaned, 'Wordcloud untuk Jawaban Permintaan Ekspor')

                    st.header('Wordclouds dari Data Tambahan:')
                    for column, jawaban in jawaban_additional_data.items():
                        st.subheader(f'Wordcloud untuk {column}:')
                        jawaban_cleaned = preprocess_text(jawaban)
                        generate_wordcloud(jawaban_cleaned, f'Wordcloud untuk {column}')

                    st.header('Analisis TF-IDF dari Jawaban Dijabarkan:')
                    st.subheader('TF-IDF untuk Jawaban Permintaan Domestik:')
                    corpus_domestik = [jawaban_domestik_cleaned]
                    tfidf_matrix_domestik, feature_names_domestik = calculate_tfidf(corpus_domestik)
                    display_top_tfidf_words(tfidf_matrix_domestik, feature_names_domestik)

                    st.subheader('TF-IDF untuk Jawaban Permintaan Ekspor:')
                    corpus_ekspor = [jawaban_ekspor_cleaned]
                    tfidf_matrix_ekspor, feature_names_ekspor = calculate_tfidf(corpus_ekspor)
                    display_top_tfidf_words(tfidf_matrix_ekspor, feature_names_ekspor)

                    st.header('Analisis TF-IDF dari Data Tambahan:')
                    for column, jawaban in jawaban_additional_data.items():
                        st.subheader(f'TF-IDF untuk {column}:')
                        jawaban_cleaned = preprocess_text(jawaban)
                        corpus = [jawaban_cleaned]
                        tfidf_matrix, feature_names = calculate_tfidf(corpus)
                        display_top_tfidf_words(tfidf_matrix, feature_names)

                    st.header('Topik yang Dibahas dari Kolom Permintaan Domestik dan Permintaan Ekspor:')

                    st.subheader('Topik Permintaan Domestik:')
                    topik_domestik = get_factors(jawaban_domestik, "Permintaan Domestik")
                    topik_domestik_text = " ".join(topik_domestik)
                    st.write(topik_domestik_text)

                    st.subheader('Topik Permintaan Ekspor:')
                    topik_ekspor = get_factors(jawaban_ekspor, "Permintaan Ekspor")
                    topik_ekspor_text = " ".join(topik_ekspor)
                    st.write(topik_ekspor_text)

                    st.header('Topik yang Dibahas dari Data Tambahan:')
                    for column, jawaban in jawaban_additional_data.items():
                        st.subheader(f'Topik {column}:')
                        topik_additional = get_factors(jawaban, column)
                        topik_additional_text = " ".join(topik_additional)
                        st.write(topik_additional_text)

                    # Generate summaries using LLM
                    kesimpulan_domestik, kesimpulan_ekspor, kesimpulan_additional = generate_summary(
                        avg_values_1, avg_values_2, topik_domestik_text, topik_ekspor_text, jawaban_additional_data,
                        quarter_now, quarter_before, phenomenon_reason
                    )

                    st.header('Kesimpulan tentang Domestik:')
                    st.write(kesimpulan_domestik)

                    st.header('Kesimpulan tentang Ekspor:')
                    st.write(kesimpulan_ekspor)

                    st.header('Kesimpulan tentang Data Tambahan:')
                    for column, kesimpulan in kesimpulan_additional.items():
                        st.subheader(f'Kesimpulan untuk topik {column}:')
                        st.write(kesimpulan)


if __name__ == "__main__":
    main()
