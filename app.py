"""Kalp hastalığı risk tahmini — Streamlit arayüzü.

Çalıştırmak için:  streamlit run app.py

Model, analysis.ipynb'nin 10. bölümünde eğitilip model/kalp_modeli.pkl dosyasına
kaydedilir. Bu arayüz yalnızca o paketi okur; burada eğitim yapılmaz.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Sabitler
# --------------------------------------------------------------------------

MODEL_YOLU = Path(__file__).parent / "model" / "kalp_modeli.pkl"

# Doğrulanmış renk paleti (notebook'taki grafiklerle aynı)
YUZEY = "#fcfcfb"
MUREKKEP = "#0b0b0b"
MUREKKEP_2 = "#52514e"
SOLUK = "#898781"
CIZGI = "#e1e0d9"
IYI = "#0ca30c"       # durum: düşük risk
KRITIK = "#d03b3b"    # durum: yüksek risk
MAVI = "#2a78d6"

# Arayüzdeki seçenek etiketleri -> veri setindeki sayısal kodlar
CP_SECENEK = {
    "Tipik anjina": 1,
    "Atipik anjina": 2,
    "Anjina dışı ağrı": 3,
    "Asemptomatik (göğüs ağrısı şikâyeti yok)": 4,
}
RESTECG_SECENEK = {
    "Normal": 0,
    "ST-T dalga anormalliği": 1,
    "Sol ventrikül hipertrofisi": 2,
}
SLOPE_SECENEK = {
    "Yukarı eğimli": 1,
    "Düz": 2,
    "Aşağı eğimli": 3,
}
THAL_SECENEK = {
    "Normal": 3,
    "Sabit defekt": 6,
    "Geri dönüşlü defekt": 7,
}

# Katkı dökümünde gösterilecek okunabilir adlar
OKUNAKLI = {
    "age": "Yaş",
    "trestbps": "Dinlenme kan basıncı",
    "chol": "Kolesterol",
    "thalach": "Maksimum kalp atış hızı",
    "oldpeak": "ST çökmesi (oldpeak)",
    "ca": "Boyanan damar sayısı",
    "sex": "Cinsiyet (erkek)",
    "fbs": "Açlık kan şekeri > 120",
    "exang": "Egzersize bağlı anjina",
    "cp_1": "Göğüs ağrısı: tipik anjina",
    "cp_2": "Göğüs ağrısı: atipik anjina",
    "cp_3": "Göğüs ağrısı: anjina dışı",
    "cp_4": "Göğüs ağrısı: asemptomatik",
    "restecg_0": "EKG: normal",
    "restecg_1": "EKG: ST-T anormalliği",
    "restecg_2": "EKG: sol ventrikül hipertrofisi",
    "slope_1": "ST eğimi: yukarı",
    "slope_2": "ST eğimi: düz",
    "slope_3": "ST eğimi: aşağı",
    "thal_3": "Talyum: normal",
    "thal_6": "Talyum: sabit defekt",
    "thal_7": "Talyum: geri dönüşlü defekt",
}


# --------------------------------------------------------------------------
# Sayfa ayarları ve stil
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Kalp Hastalığı Risk Tahmini",
    page_icon="❤️",
    layout="centered",
)

st.markdown(
    f"""
    <style>
      .blok-baslik {{
        font-size: 0.78rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: {SOLUK};
        margin: 0.4rem 0 0.2rem 0;
      }}
      .sonuc-karti {{
        background: {YUZEY}; border-radius: 14px; padding: 1.4rem 1.6rem;
        margin: 0.4rem 0 1.1rem 0; border: 1px solid {CIZGI};
      }}
      .sonuc-etiket {{
        font-size: 0.78rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; color: {SOLUK}; margin-bottom: .35rem;
      }}
      .sonuc-basligi {{ font-size: 1.65rem; font-weight: 700; line-height: 1.25; }}
      .sonuc-aciklama {{ font-size: 0.95rem; color: {MUREKKEP_2}; margin-top: .45rem; }}
      .olcer-kutu {{ margin-top: 1.1rem; position: relative; }}
      .olcer-yol {{ height: 14px; border-radius: 7px; overflow: hidden; }}
      .olcer-dolgu {{ height: 100%; border-radius: 7px; }}
      .olcer-esik {{
        position: absolute; top: -4px; height: 22px; width: 2px;
        background: {SOLUK}; left: 50%;
      }}
      .olcer-alt {{
        display: flex; justify-content: space-between;
        font-size: 0.78rem; color: {SOLUK}; margin-top: .3rem;
      }}
      .katki-satir {{
        display: flex; justify-content: space-between; gap: 1rem;
        font-size: 0.9rem; padding: .28rem 0; border-bottom: 1px solid {CIZGI};
        color: {MUREKKEP_2};
      }}
      .katki-deger {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
      .dipnot {{ font-size: 0.82rem; color: {SOLUK}; line-height: 1.5; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model yükleme
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Model yükleniyor…")
def model_paketini_yukle(yol: Path) -> dict:
    """Eğitilmiş modeli ve yanındaki meta bilgiyi diskten okur."""
    return joblib.load(yol)


if not MODEL_YOLU.exists():
    st.error(
        f"Model dosyası bulunamadı: `{MODEL_YOLU}`\n\n"
        "Önce `analysis.ipynb` dosyasını baştan sona çalıştırın — 10. bölüm modeli "
        "kaydediyor."
    )
    st.stop()

paket = model_paketini_yukle(MODEL_YOLU)
model = paket["model"]
medyan = paket["medyanlar"]
ESIK = paket.get("esik", 0.50)


# --------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# --------------------------------------------------------------------------

def ozellik_satiri_olustur(girdi: dict) -> pd.DataFrame:
    """Ham form girdisini modelin beklediği 22 sütunluk tek satıra çevirir.

    Eğitimdeki One-Hot dönüşümünün aynısını uygular, ardından sütunları
    eğitimdeki ada ve sıraya hizalar. Tek hastada yalnızca bir kategori dolu
    olacağı için kalan One-Hot sütunları 0 ile doldurulur.
    """
    ham = pd.DataFrame([girdi])
    kodlu = pd.get_dummies(
        ham, columns=paket["kategorik"], prefix=paket["kategorik"], dtype=int
    )
    return kodlu.reindex(columns=paket["ozellikler"], fill_value=0)


def katkilari_hesapla(satir: pd.DataFrame) -> pd.Series:
    """Lojistik regresyonun karar dökümü: katkı = katsayı × ölçeklenmiş değer."""
    hazirlik = model.named_steps["hazirlik"]
    katsayilar = model.named_steps["model"].coef_[0]
    adlar = [a.split("__")[-1] for a in hazirlik.get_feature_names_out()]
    olcekli = hazirlik.transform(satir)[0]
    return pd.Series(katsayilar * olcekli, index=adlar).sort_values()


def olcer_ciz(oran: float, renk: str) -> None:
    """Olasılığı gösteren ölçer: dolgu durum rengi, yol aynı rengin açık tonu."""
    st.markdown(
        f"""
        <div class="olcer-kutu">
          <div class="olcer-yol" style="background:{renk}26;">
            <div class="olcer-dolgu" style="width:{oran * 100:.1f}%; background:{renk};"></div>
          </div>
          <div class="olcer-esik"></div>
          <div class="olcer-alt">
            <span>%0</span><span>karar eşiği %50</span><span>%100</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Başlık
# --------------------------------------------------------------------------

st.title("❤️ Kalp Hastalığı Risk Tahmini")
st.caption(
    f"{paket['model_adi']} modeli · UCI Cleveland veri seti (303 hasta) · "
    f"test doğruluğu %{paket['metrikler']['test_accuracy'] * 100:.1f}"
)

st.info(
    "Bu bir **eğitim projesidir, tıbbi teşhis aracı değildir.** Sonuçlar hiçbir "
    "şekilde klinik karar için kullanılamaz. Sağlığınızla ilgili endişeniz varsa "
    "bir hekime başvurun.",
    icon="ℹ️",
)


# --------------------------------------------------------------------------
# Giriş formu
# --------------------------------------------------------------------------

with st.form("hasta_formu"):
    st.markdown('<div class="blok-baslik">Temel bilgiler</div>', unsafe_allow_html=True)

    sut1, sut2 = st.columns(2)
    with sut1:
        yas = st.slider("Yaş", 20, 90, int(medyan["age"]))
        cinsiyet = st.radio("Cinsiyet", ["Erkek", "Kadın"], horizontal=True)
        kan_basinci = st.slider(
            "Dinlenme kan basıncı (mm Hg)", 80, 220, int(medyan["trestbps"])
        )
    with sut2:
        kolesterol = st.slider("Kolesterol (mg/dl)", 100, 600, int(medyan["chol"]))
        gogus_agrisi = st.selectbox("Göğüs ağrısı tipi", list(CP_SECENEK.keys()), index=3)

    st.markdown(
        '<div class="blok-baslik">Klinik test sonuçları</div>', unsafe_allow_html=True
    )
    st.caption(
        "Bu alanlar efor testi ve görüntüleme bulgularıdır. Değerleri bilmiyorsanız "
        "varsayılanları olduğu gibi bırakabilirsiniz — varsayılanlar veri setinin "
        "medyan değerleridir."
    )

    sut3, sut4 = st.columns(2)
    with sut3:
        maks_nabiz = st.slider(
            "Egzersizde ulaşılan maks. kalp hızı", 60, 220, int(medyan["thalach"])
        )
        st_cokmesi = st.slider(
            "ST çökmesi (oldpeak)", 0.0, 7.0, float(medyan["oldpeak"]), step=0.1
        )
        damar_sayisi = st.selectbox(
            "Floroskopide boyanan damar sayısı", [0, 1, 2, 3], index=int(medyan["ca"])
        )
        aclik_sekeri = st.checkbox("Açlık kan şekeri > 120 mg/dl")
    with sut4:
        ekg = st.selectbox("Dinlenme EKG sonucu", list(RESTECG_SECENEK.keys()))
        st_egimi = st.selectbox("Egzersiz ST segmentinin eğimi", list(SLOPE_SECENEK.keys()))
        talyum = st.selectbox("Talyum stres testi", list(THAL_SECENEK.keys()))
        egzersiz_anjinasi = st.checkbox("Egzersize bağlı anjina var")

    gonder = st.form_submit_button("Tahmin Et", type="primary", width="stretch")


# --------------------------------------------------------------------------
# Tahmin
# --------------------------------------------------------------------------

if gonder:
    girdi = {
        "age": float(yas),
        "trestbps": float(kan_basinci),
        "chol": float(kolesterol),
        "thalach": float(maks_nabiz),
        "oldpeak": float(st_cokmesi),
        "ca": float(damar_sayisi),
        "sex": 1 if cinsiyet == "Erkek" else 0,
        "fbs": 1 if aclik_sekeri else 0,
        "exang": 1 if egzersiz_anjinasi else 0,
        "cp": CP_SECENEK[gogus_agrisi],
        "restecg": RESTECG_SECENEK[ekg],
        "slope": SLOPE_SECENEK[st_egimi],
        "thal": THAL_SECENEK[talyum],
    }

    satir = ozellik_satiri_olustur(girdi)
    olasilik = float(model.predict_proba(satir)[0, 1])
    riskli = olasilik >= ESIK

    if riskli:
        renk, basligi, ikon = KRITIK, "Kalp Hastalığı Riski YÜKSEK", "⚠️"
        aciklama = (
            "Girilen bulgular, veri setindeki kalp hastası profiline benziyor. "
            "Bu bir teşhis değildir; bir kardiyoloğa başvurulması önerilir."
        )
    else:
        renk, basligi, ikon = IYI, "Düşük Risk — Sağlıklı", "✅"
        aciklama = (
            "Girilen bulgular, veri setindeki sağlıklı kişi profiline benziyor. "
            "Bu, kalp hastalığı olmadığını kanıtlamaz."
        )

    st.markdown(
        f"""
        <div class="sonuc-karti" style="border-left: 5px solid {renk};">
          <div class="sonuc-etiket">Tahmin sonucu</div>
          <div class="sonuc-basligi" style="color:{renk};">{ikon} {basligi}</div>
          <div class="sonuc-aciklama">{aciklama}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sol, sag = st.columns([2, 1])
    with sol:
        st.markdown("**Modelin tahmin ettiği hastalık olasılığı**")
        olcer_ciz(olasilik, renk)
    with sag:
        st.metric("Olasılık", f"%{olasilik * 100:.1f}")

    # --- Karar dökümü -----------------------------------------------------
    with st.expander("Model bu kararı neden verdi?"):
        st.caption(
            "Lojistik regresyon doğrusal bir modeldir, bu yüzden her değişkenin karara "
            "katkısı tek tek hesaplanabilir: **katkı = katsayı × ölçeklenmiş değer**. "
            "Pozitif değerler riski artırır, negatif değerler azaltır."
        )
        katki = katkilari_hesapla(satir)
        katki = katki[katki.abs() > 0.01]

        kol1, kol2 = st.columns(2)
        with kol1:
            st.markdown(f"<b style='color:{KRITIK};'>Riski artıranlar</b>",
                        unsafe_allow_html=True)
            artiranlar = katki[katki > 0].sort_values(ascending=False).head(5)
            if artiranlar.empty:
                st.caption("Riski artıran belirgin bir bulgu yok.")
            for ad, deger in artiranlar.items():
                st.markdown(
                    f"<div class='katki-satir'><span>{OKUNAKLI.get(ad, ad)}</span>"
                    f"<span class='katki-deger'>+{deger:.2f}</span></div>",
                    unsafe_allow_html=True,
                )
        with kol2:
            st.markdown(f"<b style='color:{MAVI};'>Riski azaltanlar</b>",
                        unsafe_allow_html=True)
            azaltanlar = katki[katki < 0].head(5)
            if azaltanlar.empty:
                st.caption("Riski azaltan belirgin bir bulgu yok.")
            for ad, deger in azaltanlar.items():
                st.markdown(
                    f"<div class='katki-satir'><span>{OKUNAKLI.get(ad, ad)}</span>"
                    f"<span class='katki-deger'>{deger:.2f}</span></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Modele girilen değerler (ham veri)"):
        st.dataframe(pd.DataFrame([girdi]), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Alt bilgi
# --------------------------------------------------------------------------

st.divider()

with st.expander("Model hakkında ve bilinen sınırlılıkları"):
    m = paket["metrikler"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Test doğruluğu", f"%{m['test_accuracy'] * 100:.1f}")
    k2.metric("Duyarlılık (recall)", f"%{m['test_recall'] * 100:.1f}")
    k3.metric("Kesinlik (precision)", f"%{m['test_precision'] * 100:.1f}")

    st.markdown(
        f"""
        **Model:** {paket['model_adi']} · 5 katlı çapraz doğrulama doğruluğu
        **%{m['cv_accuracy'] * 100:.1f}** · eğitim tarihi {paket['egitim_tarihi']}
        · scikit-learn {paket['surumler']['sklearn']}

        **Bilinen sınırlılıklar:**

        - Model 1988 tarihli, **303 hastalık** tek merkezli (Cleveland) bir veri setiyle
          eğitildi. Günümüz hasta popülasyonunu temsil etmeyebilir.
        - Veri seti bir **klinik sevk popülasyonundan** gelir; bu kişilerin tamamı zaten
          bir nedenle teste yönlendirilmiştir. Bu yüzden genel topluma uygulanamaz.
        - **Kadın hastalarda daha zayıf çalışır.** Veri setinde 206 erkeğe karşılık 97
          kadın var ve model, cinsiyete en büyük katsayıyı atamış durumda. Hata
          analizinde, aynı klinik bulgulara sahip bir hastanın kadın olduğu için
          kaçırıldığı bir vaka tespit edildi.
        - Karar eşiği %50'dir. Tıbbi tarama bağlamında eşiği düşürmek, daha fazla yanlış
          alarm karşılığında daha az hasta kaçırmak anlamına gelir.
        """
    )

st.markdown(
    '<div class="dipnot">Bu uygulama bir makine öğrenmesi dersi projesidir. '
    'Tıbbi tavsiye, teşhis veya tedavi yerine geçmez.</div>',
    unsafe_allow_html=True,
)
