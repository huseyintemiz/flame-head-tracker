# SUDFace Analiz Notlari

## Ornek video: `mov_subj11_NA`

Bu not, yerel SUDFace annotation dosyalarindan cikarilan ornek bir video analizini saklar.

Kullanilan kaynaklar:

- `SUDFace/SUDFace_28_32s_dataset_(Validation_Experiment2-Middle).xlsx`
- `SUDFace/Expression Changes.xlsx`

## Clip-level insan degerlendirmeleri

Video:

- `mov_subj11_NA`

Clip:

- `middle (28-32s)`

Ozet istatistikler:

- `neutralness`: `n=33`, ortalama `4.06`, medyan `4`, aralik `1..7`
- `naturalness`: `n=33`, ortalama `4.91`, medyan `5`, aralik `1..7`
- `valence`: `n=33`, ortalama `0.61`, medyan `1`, aralik `-3..3`

Mental state dagilimi:

- `Concentrated`: `11`
- `Relaxed`: `10`
- `Thinking`: `5`
- `Proud`: `3`
- `Bored`: `2`
- `Stressed`: `1`
- `Confused`: `1`

## Expression-change interval annotation

Ayni video icin `Expression Changes.xlsx` kaydi:

- `rater`: `Assistant 2`
- `onset`: `34`
- `offset`: `37`
- `direction`: `positive`

## Annotator bilgisi

SUDFace icinde annotator bilgisi iki farkli sekilde tutuluyor.

### Validation clip rating annotator'lari

Validation spreadsheet'lerinde annotator'lar acik isimle verilmemis. Bunlar anonim katilimcilar ve ancak survey icindeki `Custom Participant ID` alani ile ayirt edilebiliyor.

Yerelde guvenilir sekilde sayabildigim dosya:

- `SUDFace_28_32s_dataset_(Validation_Experiment2-Middle).xlsx`

Bu dosyada:

- toplam `33` farkli annotator var
- `150` clip'in her biri `33` farkli annotator tarafindan degerlendirilmis

Yani `middle (28-32s)` clip'leri icin:

- bir clip basina annotator sayisi: `33`

Not:

- `SUDFace_7_11s_dataset_(Validation_Experiment1-Begining).xlsx` dosyasi da benzer survey export yapisina sahip, ancak yerel export semasi daha kirli oldugu icin bu dosya icin ayni sayi burada kesin bilgi olarak verilmemistir.

### Expression-change annotator'lari

`Expression Changes.xlsx` dosyasinda annotator'lar acikca yaziyor.

Gorulen annotator isimleri:

- `Assistant 1`
- `Assistant 2`

Bu dosyada video bazinda annotator sayisi:

- bazi videolarda `1`
- bazi videolarda `2`

`mov_subj11_NA` icin expression-change annotation:

- yalnizca `Assistant 2`

## Yorum

Bu sonuc, `mov_subj11_NA` klibinin genel olarak beklenen sekilde notr ve dogal algilandigini, ancak tamamen duygusuz okunmadigini gosteriyor.

`neutralness` ortalamasi `4.06/7` oldugu icin klip orta-ust duzeyde notr algilanmis, fakat cok guclu bicimde tam notr degil. `naturalness` ortalamasi `4.91/7` ile daha yuksek; bu da ifadenin yapay ya da zorlanmis degil, gorece dogal bulundugunu gosteriyor. `valence` ortalamasinin `0.61` olmasi, klibin sifira cok yakin ama hafif pozitif yone egimli algilandigini dusunduruyor.

Mental state dagilimi da bunu destekliyor. En baskin iki etiket `Concentrated` ve `Relaxed`. Bu, ifadenin belirgin bir duygu sergilemekten cok kontrollu, sakin ve dikkat odakli bir yuz olarak algilandigini gosteriyor. `Stressed` ve `Confused` etiketlerinin cok az secilmesi, klibin guclu negatif veya karisik bir ifade vermedigine isaret ediyor.

`34-37` araligindaki `positive` yonlu expression-change kaydi, hafif pozitif valence ile uyumlu olabilir. Bu, videonun belirli bir kisminda kisa sureli mikro-pozitif bir ifade kaymasi oldugunu dusundurur. Bu nedenle video notr veri olarak kullanilabilir, ancak cok kati notrluk gerektiren analizlerde bu aralik ayri ele alinabilir.

## Kisa sonuc

- Video notr veri olarak kullanilabilir.
- Ifade genel olarak dogal gorunuyor.
- Tam anlamiyla kusursuz notr bir ornek degil.
- `34-37` araligi, daha siki filtreleme gereken durumlarda dikkat edilmesi gereken kisimdir.
