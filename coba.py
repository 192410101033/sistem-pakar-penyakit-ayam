import operator
#import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
#from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
#from streamlit.components.v1 import html

st.set_page_config(page_title="Sistem Pakar", page_icon=":chicken:", layout="wide")


class FuzzyKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5, plot=False):
        self.k = k
        self.plot = plot

    def fit(self, X, y=None):
        self._check_params(X, y)
        self.X = X
        self.y = y
        self.xdim = len(self.X[0])
        self.n = len(y)
        classes = list(set(y))
        classes.sort()
        self.classes = classes
        self.df = pd.DataFrame(self.X)
        self.df['y'] = self.y
        self.memberships = self._compute_memberships()
        self.df['membership'] = self.memberships
        self.fitted_ = True
        return self

    def predict(self, X):
        if self.fitted_ is None:
            raise Exception('predict() called before fit()')
        else:
            m = 2
            y_pred = []
            self.top_memberships = []
            top_memberships = []  # Menyimpan 5 nilai keanggotaan terbesar untuk ditampilkan
            for x in X:
                neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
                votes = {}
                memberships = {}  # Menyimpan nilai keanggotaan untuk setiap kelas
                for c in self.classes:
                    den = 0
                    for n in range(self.k):
                        dist = np.linalg.norm(x - neighbors.iloc[n, 0:self.xdim])
                        den += 1 / (dist ** (2 / (m - 1)))
                    neighbors_votes = []
                    for n in range(self.k):
                        dist = np.linalg.norm(x - neighbors.iloc[n, 0:self.xdim])
                        num = (neighbors.iloc[n].membership[c]) / (dist ** (2 / (m - 1)))
                        vote = num / den
                        neighbors_votes.append(vote)
                    votes[c] = np.sum(neighbors_votes)
                    memberships[c] = np.sum(neighbors_votes)  # Simpan nilai keanggotaan

                # Menyimpan 5 nilai keanggotaan terbesar
                top_5_memberships = sorted(memberships.items(), key=lambda x: x[1], reverse=True)[:5]
                self.top_memberships.append(top_5_memberships)

                pred = max(votes.items(), key=operator.itemgetter(1))[0]
                y_pred.append(pred)

            # Menampilkan 5 nilai keanggotaan terbesar untuk setiap data uji
            for i, top_5 in enumerate(top_memberships):
                st.write(f"\n5 Nilai Keanggotaan Terbesar pada Data Uji ke-{i + 1}:")
                for c, value in top_5:
                    st.write(f"{c}: {value}")

            return y_pred       

    def _find_k_nearest_neighbors(self, df, x):
        X = df.iloc[:, 0:self.xdim].values
        df['distances'] = [np.linalg.norm(X[i] - x) for i in range(self.n)]
        df.sort_values(by='distances', ascending=True, inplace=True)
        neighbors = df.iloc[0:self.k]
        return neighbors

    def _get_counts(self, neighbors):
        groups = neighbors.groupby('y')
        counts = {group[1]['y'].iloc[0]: group[1].count()[0] for group in groups}
        return counts

    def _compute_memberships(self):
        memberships = []
        for i in range(self.n):
            x = self.X[i]
            y = self.y[i]
            neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
            counts = self._get_counts(neighbors)
            membership = dict()
            for c in self.classes:
                try:
                    uci = 0.49 * (counts[c] / self.k)
                    if c == y:
                        uci += 0.51
                    membership[c] = uci
                except:
                    membership[c] = 0
            memberships.append(membership)
        return memberships

    def _check_params(self, X, y):
        if type(self.k) != int:
            raise Exception('"k" should have type int')
        if self.k >= len(y):
            raise Exception('"k" should be less than the number of feature sets')
        if self.k % 2 == 0:
            raise Exception('"k" should be odd')
        if type(self.plot) != bool:
            raise Exception('"plot" should have type bool')
    


# Memuat data latih dari file CSV
train_data = pd.read_csv('data_setfix.csv')
train_features = train_data.iloc[:, :-1].values
train_labels = train_data.iloc[:, -1].values

# Mendapatkan nama kolom/gejala dari data latih
column_names = train_data.columns.tolist()

# Solusi atau saran untuk setiap penyakit berdasarkan hasil prediksi
solutions = {
    'Salesma Ayam / Infectious coryza / SNOT': '1.	Menggunakan sulfadimethoxine dengan dicampurkan air minum ayam tersebut\n2. Pemakaian antibiotik menggunakan tylosin, tetracycline, erythromycin, dan spectinomycin',
    'Berak Kapur / Pullorum disease' : '1.	Usaha terbaik menyuntikan antibiotik seperti, neo terramycin dan cocillin namun tidak efektif untuk menghilangkan penyakit namun mencegah kematian',
    'Flu Burung / Avian influenza': '1.	Usaha terbaik membuat kondisi ayam membaik dan merangsang nafsu makannya dengan pemberian vitamin dan mineral, serta memberikan antibiotik. Selain itu pemanasan tambahan pada kandang menjadi salah satu pengobatan pada penyakit flu burung.',
    'Berak Darah / Coccidiosis': '1.	Menggunakan obat yang bersifat coccidiostat atau coccidioidal\n2.	Menggunakan obat seperti asam folat antagonis, amprolium, halofuginone, hydrobromide, ionophore, nicarbazine, nitrobenzamide, clopidol, robenidine dan sulfaquinoxalin\n3.	Obat tersebut dicampurkan kedalam pakan atau air minum',
    'Kolera Ayam / Fowl cholera': '1.	Memberikan pengobatan dengan antimikroba seperti preparat sulfa antara lain sulfaquinoxalin, sulfamethazine, dan sulfamerazin diberikan dengan media pakan dan air minum\n2.	Antimikroba yang lain seperti antibioka dengan memberikan streptomycin, dan terramisin',
    'Ngorok / Chroic respiratory desease (CRD)': '1.	Pengobatan permulaan diberikan obat seperti linkomisin, spektinomisi, streptomycin, oxytetracyclin, spiramycin, tylosin, dan golongan kuinolon seperti norflosasin dan enrofloksasin.\n2.	Selanjutnya pemberian vitamin',
    'Tetelo / Newcastle desease': '1.	Usaha terbaik membuat kondisi ayam membaik dan merangsang nafsu makannya dengan pemberian vitamin dan mineral, serta memberikan antibiotik. Selain itu pemanasan tambahan pada kandang menjadi salah satu pengobatan pada penyakit tetelo.',
    'Gumboro / Infectious bursal disease': '1.	Usaha terbaik memberikan pengobatan tetes 5%, gula merah 2% dicampur dengan NaHC03 0,2%\n2.	Pemberian mineral, vitamin, dan elektrolit\n3.	Pemberian antibiotik dan mengurangi kadar protein dalam makanan',
    'Batuk Ayam Menahun / Infectious bronchitis / IB': '1.	Usaha terbaik membuat kondisi ayam membaik dan merangsang nafsu makannya dengan pemberian vitamin dan mineral, serta memberikan antibiotik. Selain itu pemanasan tambahan pada kandang menjadi salah satu pengobatan pada penyakit batuk ayam menahun.',
    'Necrotic Enteritic': '1.	Pemberian obat seperti Fithera\n2.	Pemberian multivitamin yang mengandung vitamin A dan K\n3.	Pemberian produk yang mengandung asam amino seperti Aminovit',
    'Batuk Darah / Infectious laryngotracheitis': '1.	Usaha terbaik membuat kondisi ayam membaik dan merangsang nafsu makannya dengan pemberian vitamin dan mineral, serta memberikan antibiotik. ',
    'Kolibasilosis / Colibacillosis': '1.	Pemberian obat seperti Ampicol, Collimezyn, atau Neo Meditril\n2.	Pemberian multivitamin seperti Vita Stress atau Fortevit\n3.	Pemberian sanitasi pada air minum dengan Desinsep atau Medisep  ',
}
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()


lottie_animation_1 = load_lottie_url("https://lottie.host/a127ad45-8c6a-44a9-8e52-b9fc4c5dc238/9Ou9sbfeDc.json")

with st.sidebar:
    selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Diagnosa", "Jenis Penyakit"],
            icons=["house", "hospital", "capsule-pill"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
)

if selected == "Home":
    right_column, left_column = st.columns(2)

    with right_column:
        st.write("""
        # Sistem Pakar Diagnosa Penyakit Ayam
        **Sistem Pakar Diagnosa Penyakit Ayam Menggunakan Metode Fuzzy KNN Studi Kasus Peternakan PT. Cemerlang Unggas Lestari**
        """)
        st.write("##")
    with left_column:
        st_lottie(lottie_animation_1, height=320)

    st.write("---")
    st.subheader("Tentang Sistem")
    st.markdown("""
        <div style="text-align: justify;">
            Platform online yang dirancang untuk membantu peternak ayam pedaging dalam mengidentifikasi dan mendiagnosis penyakit yang mungkin menyerang ayam-ayam mereka. Website ini menggunakan teknologi sistem pakar yang memanfaatkan pengetahuan dan pengalaman dari ahli dokter hewan atau pakar kesehatan hewan dalam bidang diagnosa penyakit ayam. Sistem ini menggunakan Metode FKNN yang dapat mengatasi ketidakpastian atau ketidakjelasan pada data, seperti penyakit yang bermacam-macam namun gejalanya tidak jauh berbeda. Dengan menggunakan fuzzy logic sebagai dasar pengambilan keputusan, FK-NN mampu memberikan hasil diagnosa yang lebih akurat dalam pengambilan keputusannya. Cukup klik tombol diagnosa pada bar navigasi selanjutnya centang gejala pada kolom centang sesuai dengan gejala yang diderita ayam dan klik tombol diagnosa akan muncul hasil diagnosa seperti nama penyakit, solusi, hingga hasil perhitungan dari metode FKNN.
        </div>
    """, unsafe_allow_html=True)
    st.write("**Dataset diperoleh dari Dinas Peternakan dan Kesehatan Hewan Provinsi Jawa Tengah dengan total 150 data**")



if selected == "Diagnosa":
    st.title(f"Menu yang dipilih {selected}")
    st.write("""
    # Diagnosa Penyakit
    Pilih gejala dengan centang checkbox sesuai dengan kondisi ayam yang diternak
    """)

    # Membagi gejala menjadi beberapa kolom dalam satu baris
    num_columns = 3
    gejala = [column_names[i:i+num_columns] for i in range(0, len(column_names)-1, num_columns)]

        # Function to reset checkbox states
    def reset_checkboxes():
        st.session_state.checkbox_states = {col: False for col in column_names[:-1]}

    # Initialize checkbox states in session state
    if 'checkbox_states' not in st.session_state:
        reset_checkboxes()

    # Membuat inputan gejala menggunakan checkbox dengan nama kolom yang sesuai
    input_data = []
    for col_group in gejala:
        cols = st.columns(len(col_group))
        for i, col_name in enumerate(col_group):
            state_key = f"{col_name}_checkbox"
            checked = cols[i].checkbox(col_name, key=state_key, value=st.session_state.checkbox_states.get(col_name, False))
            input_data.append(checked)
            st.session_state.checkbox_states[col_name] = checked

    # Konversi inputan gejala ke array numpy
    input_array = np.array(input_data, dtype=np.float64).reshape(1, -1)

    # Melatih classifier Fuzzy k-NN
    k = 5
    classifier = FuzzyKNN(k=k)
    classifier.fit(train_features, train_labels)

    st.write("**Klik 2 kali tombol reset untuk mereset centang pada checkbox*")

    if st.button('Reset'):    
        reset_checkboxes()

    if st.button('Diagnosa', type="primary"):
        # Menguji classifier menggunakan data uji
        predictions = classifier.predict(input_array)
        
        # Menampilkan hasil prediksi
        result = predictions[0]
        st.subheader(f"Hasil diagnosa untuk gejala yang dimasukkan: Diagnosa = {result}")

        # Menampilkan solusi atau saran berdasarkan hasil prediksi
        if result in solutions:
            st.write("---")
            st.subheader("\nSolusi atau Saran:")
            st.info(solutions[result])
        else:
            st.subheader("\nTidak ditemukan solusi untuk penyakit yang didiagnosa.")
        
        # Menampilkan tabel gejala yang dicentang saat menekan tombol prediksi
        checked_symptoms = [column_names[i] for i in range(len(column_names) - 1) if input_array[0][i] == 1]
        if checked_symptoms:
            st.write("---")
            st.subheader("\nGejala yang Dicentang:")
            for i, symptom in enumerate(checked_symptoms, start=1):
                st.info(f"{i}. {symptom}")
        else:
            st.subheader("\nTidak ada gejala yang dicentang.")
            

        for i, top_5 in enumerate(classifier.top_memberships):
            st.write("---")
            st.subheader(f"\n5 Nilai Keanggotaan Terbesar :")
            st.write("*Catatan : semakin mendekati nilai 1 semakin akurat*")
            for c, value in top_5:
                st.info(f"{c}: {value:.2f}")


        neighbors = classifier._find_k_nearest_neighbors(pd.DataFrame.copy(classifier.df), input_array.flatten())
        st.write("---")
        st.subheader("\n5 Nilai Jarak Euclidean Terdekat :")
        for index, row in neighbors.iterrows():
            # Calculate the membership value in percentage
            membership = (row['distances'])
            st.info(f"{row['y']}: {membership:.2f}")

if selected == "Jenis Penyakit":
    st.title(f"Menu yang dipilih {selected}")
    st.write("""
    # Jenis Penyakit
    Daftar Nama Penyakit Ayam Pedaging beserta Gejala, Cara Pengobatan, dan Pencegahan
    """)
    data = pd.read_csv('jenis penyakit.csv')

    render_image = JsCode('''
                        
        function renderImage(params) {
        // Create a new image element
        var img = new Image();

        // Set the src property to the value of the cell (should be a URL pointing to an image)
        img.src = params.value;

        // Set the width and height of the image to 50 pixels
        img.width = 50;
        img.height = 50;

        // Return the image element
        return img;
        }
    '''
    )
    # Build GridOptions object
    options_builder = GridOptionsBuilder.from_dataframe(data)
    options_builder.configure_column('Gambar Penyakit', cellRenderer = render_image)
    options_builder.configure_selection(selection_mode="single", use_checkbox=True)
    grid_options = options_builder.build()

    # Create AgGrid component
    grid = AgGrid(data, 
                    gridOptions = grid_options,
                    allow_unsafe_jscode=True,
                    height=200, width=500, theme='streamlit')
    
    sel_row = grid["selected_rows"]
    if sel_row:
        with st.expander("Selections", expanded=True):
            col1,col2 = st.columns(2)
            st.subheader("Pengertian Penyakit :")
            st.info(sel_row[0]['Pengertian Penyakit'])
            st.subheader("Gejala Penyakit :")
            st.info(sel_row[0]['Gejala Penyakit'])
            st.subheader("Cara Pengobatan :")
            st.info(sel_row[0]['Cara Pengobatan'])
            st.subheader("Cara Pencegahan :") 
            st.info(sel_row[0]['Cara Pencegahan'])               
            col1.image(sel_row[0]['Gambar Penyakit'],caption=sel_row[0]['Nama Penyakit'])

