# Tezin hikâyesi (hocaya anlatmak için)

Bu dosya basit Türkçe ile yazılmıştır, teknik terimler İngilizce bırakılmıştır. Her adımda dört kutu vardır:
**Ne yaptım / Neden / Ne çıktı / Ne anlama geliyor.** Rakamlar `docs/CREDIT_*.md` ve `results/v2/*.csv` dosyalarından alınmıştır.
Durum: 21 Ekim 2026. "⏳" = deney koşuyor, sonuç henüz yok.

## Tek cümlelik hikâye
Standart conformal prediction (CP) **ortalamada** %90 coverage verir ama bir grup (ör. kadınlar) için %80 olabilir.
AFCP (Zhou & Sesia) bu grubu kendisi bulup düzeltmeyi vaat eder. Biz kredi verisinde, **kontrollü bias koyarak**,
AFCP'nin bunu ne zaman bulduğunu ve düzeltmelerin neden bazen işe yaramadığını test ettik.

---

## Adım 0 — Pilot (Temmuz)
**Ne yaptım.** İki dataset (South German Credit, Give Me Some Credit), 5 yöntem (Marginal, Partial, Exhaustive,
AFCP-Adaptive, Adaptive1), 5 seed. Pilotun 5 deneyi:
1. SGC, age ≤25 vs >25
2. SGC, age median split (≤33)  → yaş sınırı sonucu değiştirdi (group-definition sensitivity)
3. GMSC, age 3 grup (≤35 / 36–55 / >55)
4. GMSC sample-size sweep (train=cal birlikte 2000→8000): 5 seed'in ikisinde (2023, 2027) AFCP 4000'den itibaren age'i seçti, üçünde hiç seçmedi. Train ve calibration birlikte arttığı için karışık (confounded).
5. GMSC multi-attribute (age+income+dependents): final seçim hep boş; sebep: label 0 için age, label 1 için income seçiliyor, kod **kesişim** alıyor (label veto).

**Neden.** Zhou & Sesia'nın yöntemini kredi verisinde denemek.
**Ne çıktı.** Overall coverage ≈ %90, worst-group biraz düşük (GMSC 0.875). AFCP çoğunlukla hiçbir şey seçmedi.
**Ne anlama geliyor.** "AFCP çalışmıyor" değil; üç şey yüzünden test edilemedi:
1. NN 500 epoch overfit → p̂ ≈ 0/1 → score'lar aynı → t-test çalışmaz.
2. Kod sessizce `conditional=True` (label-conditional) kullanıyordu + nadir default sınıfı (%6.7) → set size 1.88, label veto.
3. Ortada **tespit edilecek bias yoktu** (temiz veride grup farkı 1–2 puan).

## Adım 1 — v2 tasarımı
**Ne yaptım.** Üç dataset: SGC (küçük), GMSC (nadir default), **Taiwan** (ana test bed: 30.000 kişi, %22 default, 4 demografik attribute).
NN'e early stopping + dropout, bir de Logistic Regression. İki calibration regime'i (marginal / label-conditional) açıkça.
10 seed (2023–2032). Split boyutları en küçük (grup × label) hücresinden türetildi.
**Neden.** Pilotun 3 sorununu tek tek gidermek.
**Ne çıktı.** Taiwan'da AUROC ≈ 0.76, set size ≈ 1.34 (marginal regime) — informative sets.
**Ne anlama geliyor.** Artık deneyin sonucu kodun sessiz varsayılanlarına değil, veriye bağlı.

## Adım 2 — Temiz veri (baseline)
**Ne yaptım.** Taiwan'da bias yokken 4 yöntem.
**Ne çıktı.** Coverage: Marginal 0.902, Partial 0.908, Exhaustive 0.906, AFCP 0.906. Kadın 0.900 / erkek 0.904 → **sex farkı yok**, AFCP sex'i hiç seçmiyor (0.00).
**Ne anlama geliyor.** Kontrol grubu temiz: AFCP boş yere alarm vermiyor. Sonraki adımdaki değişim bias'tan geliyor.
GMSC: ≤35 yaş 0.894 / 36–55 0.903 / >55 0.907. SGC: 0.89 vs 0.90, set 1.45–1.69; AFCP age'i 10 seed'in 3'ünde seçiyor.

## Adım 3 — Bias enjekte etme (iki yön)
**Ne yaptım.** Kadınların etiketlerinin ε kadarını bozdum, test hep temiz:
- **Stigma** (non-default → default): model kadınlara fazla default der.
- **Hidden defaults** (default → non-default): model kadınlardaki default'ları kaçırır.
(Zhou & Sesia bias'ı sentetik veride grubun label'ını rastgeleleştirerek kurdu; biz gerçek veride, tarihsel label hatası olarak, doz bazlı.)
**Ne çıktı (Taiwan, marginal regime).**
- Hidden defaults: kadın coverage 0.900 → 0.847 (ε .5) → **0.801 (ε .7)**, erkek ≈ 0.89–0.90; gap 0.12.
- Stigma: kadın coverage **yükseliyor** (0.900 → 0.951 at ε .3) ama set size 1.31 → 1.74. Yüksek dozda (ε .4–.5) düşüyor (non-monotone).
**Ne anlama geliyor.** İki bias farklı şekilde görünür: biri coverage'ı düşürür (hidden defaults), diğeri informativeness'i bozar (stigma).

## Adım 4 — Düzeltme yöntemleri işe yarıyor mu?
**Ne çıktı.** Calibration set de bozuksa Partial/Exhaustive/AFCP kadın coverage'ını **geri getirmiyor** (0.79–0.81, ε .7).
Calibration temizse (train-only injection, ε .5 hidden defaults): Partial/Exhaustive 0.904/0.905 (train+cal'de 0.85/0.84).
**Ne anlama geliyor.** Equalized coverage, **kalibre edildiği label'lara göre** eşitler, gerçek label'lara göre değil. Bu tezin merkez metodolojik mesajı.

## Adım 5 — Label-conditional regime
**Ne çıktı.** Hidden defaults ε .7: kadın coverage 0.86 (marginal'de 0.80), Partial/Exhaustive ≈ 0.90 geri getiriyor (kalan default'lar gerçek). Set size ≈ 1.7. AFCP seçimi zayıf (≤ 0.2) ve label 0/1 flag'leri farklı (0.20 vs 0.40) → **veto burada sınırlayıcı**.
**Ne anlama geliyor.** Label-conditional daha "adil" ama sets daha az informative (coverage vs informativeness trade-off). Pilottaki veto burada nedensel olarak görüldü.

## Adım 6 — AFCP bias'ı bulabiliyor mu? (detection)
**Ne çıktı (Taiwan, marginal).** AFCP'nin sex'i seçme oranı ε ile artıyor: hidden defaults 0 → 0.6, stigma 0 → 0.4. Seçim seed bazlı (ya hep ya hiç). ε=0'da 0.00 (yanlış alarm yok).
**GMSC'de bulgu tekrar etmiyor:** seçim oranı her dozda 0.20, hep aynı 2 seed (2023, 2027). Hipotez "calibration kirlenmesi" **desteklenmedi** (train-only'de de 0.10).
**Ne anlama geliyor.** AFCP'nin gücü veri setine bağlı; nadir default + near-singleton set'lerde test zayıf kalıyor. (⏳ `ttest_delta`/`sig_level` duyarlılığı henüz yok.)

## Adım 7 — Adaptive1 (test olmadan)
**Ne çıktı.** Taiwan: sex seçimi 0.1 / 0.6 / 0.7 (ε 0 / .3 / .7), Adaptive 0 / .4 / .6. "Hiçbirini seçme" ≈ 0. Coverage ve set size aynı.
**Ne anlama geliyor.** Test olmadan AFCP "en kötü grubu seç" kuralına dönüşüyor, dedektör değil; ε=0'da yanlış alarm (0.10).

## Adım 8 — LR ile sağlamlık
**Ne çıktı.** ε=0.2'de NN ve LR birbirine ≈ 1 puan yakın; bulgular NN'e özgü değil. Stigma ε .5 (LR): kadın 0.82 (yüksek dozda under-coverage).

## Adım 9 — Pilotun tamamının v2'de tekrarı: sample-size sweep (sonuç geldi)
Karar: pilotun tüm deneyleri v2'de tekrar edilir, teze alınacak olanlar sonra seçilir.

**Ne yaptım.** İki tasarım: (a) **joint**: train=cal birlikte 2000/4000/6000/8000 (pilot gibi, karışık); (b) **calibration-only**: train 6000 sabit, calibration 1000/2000/4000/6000. İki regime, yeni NN. GMSC (bias yok, 5 seed, test 1000) ve Taiwan (10 seed, test 500; bias yok ve hidden defaults ε .5).
**Neden.** Pilotta "veri artınca AFCP age'i seçmeye başladı" gördük ama train da büyüdüğü için neyin etkisi olduğunu bilemedik.

**Ne çıktı — GMSC (AFCP'nin age'i seçme oranı, seed bazında).**
- `conditional=True` (pilotun regime'i): seed **2023 ve 2027 her boyutta seçiyor, 2024–2026 hiçbir boyutta seçmiyor** (joint 2000'den 8000'e, calibration-only 2000–4000'de de aynı). Seçim oranı ortalama 0.4.
- Tek istisna: calibration 1000'de (train 6000) seed 2023 seçmiyor, 2000'de seçmeye başlıyor.
- `conditional=False`: seed 2027 hep seçiyor; 2023 sadece 6000/6000 ve 8000/8000'de; diğerleri hiç.
- Pilotta da aynı iki seed (2023, 2027) seçmişti.

**Ne çıktı — Taiwan.**
- Bias yokken AFCP sex'i nadiren seçiyor (0–0.2, her boyutta) → boyuta göre artan yanlış alarm yok.
- Hidden defaults ε .5 (marginal regime): seçim oranı 0.4–0.5, boyuta göre **net bir trend yok** (calibration 1000: .50, 2000: .40, 4000: .50; joint 2000: .27, 4000: .50, 8000: .40). Kadın coverage her boyutta ≈ 0.83–0.85, erkek 0.87–0.90.
- Label-conditional regime'de seçim 0.10–0.20, düz.

**Ne anlama geliyor.** Pilottaki "sample size arttıkça AFCP seçmeye başlıyor" hikâyesi bu verilerde desteklenmiyor: seçim **hangi seed'in (hangi veri örneğinin) çekildiğine** bağlı; daha fazla calibration verisi bunu belirgin değiştirmiyor (en fazla en küçük boyutta etkisi var). Pilot sweep'in karışıklığı bu yüzden sorun olmadı ama doğru tasarımla (calibration-only) bunu söyleyebiliyoruz. Taiwan'da etki zaten büyük (kadın 0.85), veri artsa da coverage düzelmiyor çünkü calibration label'ları bozuk.
Rakamlar: `results/v2/gmsc_sweep_summary.csv`, `results/v2/taiwan_sweep_summary.csv`.

## Adım 10 — GMSC çoklu attribute (age + income + dependents)

**Ne yaptım.** Pilotun "multi-attribute" deneyini GMSC'de v2 ile tekrarladım: tek attribute yerine üç aday (age 3'lü, income yüksek/düşük, dependents var/yok), Marginal / Partial / Exhaustive / AFCP / Adaptive1, 5 seed.
**Neden.** Pilotta AFCP çoklu attribute'ta "hep boş küme" seçmişti; bunun gerçek mi yoksa overfit'li NN'in eseri mi olduğunu görmek istiyorum.
**Ne çıktı.** Pilottaki "her zaman boş" davranışı **tekrarlanmadı**. Veto mekanizması hâlâ var (bir label'da bayrak çıkınca attribute elenebiliyor) ama sistematik değil.
**Ne anlama geliyor.** Pilottaki boşluk büyük ihtimalle overfit'li NN + sessiz label-conditional ayarının sonucuydu, yönteme ait bir özellik değil.
Tablolar: `results/v2/afcp/gmsc_multi/` (10 dosya).

## Adım 11 — Grup tanımı (RQ4): sonuç "grup nasıl tanımlandığına" bağlı mı?

**Ne yaptım.** Aynı veride, **bias enjekte etmeden (temiz veri)**, aynı attribute'u farklı kesimlerle tanımladım ve her tanım için tüm yöntemleri koştum: SGC age ≤25 vs ≤33; GMSC age <25 vs ≥60; Taiwan age <25 vs 3'lü yaş. Toplam 120 koşu (SGC 10 seed × nn/lr × iki regime, GMSC 5 seed, Taiwan 10 seed).
**Neden.** "Korunan grup" tez verisinde kendiliğinden belli değil; kesim noktası değişince "en kötü grup" ve yöntemlerin sıralaması değişiyor mu, bilmem lazım.
**Ne çıktı (worst-group coverage / gap; hedef 0.90).**
- **SGC (NN, marginal regime):** ≤25 → en küçük grup test'te ≈57 kişi; Marginal worst 0.871, gap 0.042; Partial 0.882; AFCP 0.873; Adaptive1 0.879. ≤33 → ≈142 kişi; Marginal 0.879, gap 0.035; Partial 0.880; AFCP 0.878; Adaptive1 0.887. Yöntemler arası fark ≤ ~0.015 → gürültü içinde. LR'de de aynı resim.
- **GMSC:** age <25 → test'te **ortalama sadece ≈8 kişi** (500'den). Marginal worst 0.803, gap 0.137; Partial/Exhaustive 0.832, gap 0.108. age ≥60 → ≈160 kişi, Marginal worst 0.893, gap 0.017. Yani "büyük adaletsizlik" sadece çok küçük grup tanımında görünüyor.
- **Taiwan:** age <25 → ≈45 kişi, Marginal worst 0.883, gap 0.029; 3'lü yaş → en küçük grup ≈14 kişi, Marginal 0.883, gap 0.060.
- **AFCP'nin seçim oranı** aynı veri/model/regime içinde tanımdan neredeyse bağımsız (SGC 0.3, GMSC 0.2/0.4, Taiwan 0.5) → seçimi tanım değil seed belirliyor (tek istisna: SGC LR conditional=True, ≤33'te 0.195).
**Ne anlama geliyor.**
1. Grup tanımı "en kötü grup" rakamını güçlü biçimde değiştiriyor — ama çoğunlukla **grup küçüldüğü için** (test n≈500'de 8–14 kişilik grupta coverage'ın standart hatası çok büyük).
2. Equalized yöntemler en küçük gruplarda Marginal'e göre ~1–3 puan iyileştirme gösteriyor (GMSC <25: 0.803→0.832) ama bu n≈8 ile güvenilir değil; tezde "küçük grup, kesin yorum yok" diye açıkça yazmalıyız.
3. Test boyutu 500 → grup coverage'ı için SE ≈ 2–4 puan; yöntem farkları bunun altında.
Tablo: `results/v2/rq4_groupdef_summary.csv`; script: `thesis_experiments/v2/analysis/group_def_table.py`.

## Araştırma soruları (taslak)
- RQ1: Label bias coverage ve set size'da nasıl görünür?
- RQ2: Equalized-coverage yöntemleri düzeltir mi? (sadece calibration label'ları temizse)
- RQ3: AFCP bias'ı ne zaman bulur, ne zaman susar? (regime, veto, dataset)
- RQ4: Sonuçlar grup tanımına duyarlı mı?
