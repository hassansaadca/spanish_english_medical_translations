### Data directory structure:
```
|--data/UFAL/
|----medical.all
|----medical.train_val
|----medical.test
```

Info from UFAL Team, copy/pasted from their README:
###################################################################
## UFAL Medical Corpus 1.0 for WMT17 Biomedical Translation Task ##
###################################################################

UFAL Medical Corpus 1.0 was prepared in the EU projects HimL (http://www.himl.eu/),
Khresmoi (http://khresmoi.eu/) and KConnect (http://k-connect.org/).
The full details describing corpus sources and pre-processing are available in
the following project deliverables:

  D1.1: Report on Building Translation Systems for Public Health Domain
  http://www.himl.eu/deliverables

  D1.2 Toolkit and Report for Translator Adaptation to New Languages
  http://kconnect.eu/eu-project-deliverables

This package contains UFAL Medical Corpus for the WMT17 Biomedical Translation Task
(http://www.statmt.org/wmt17/biomedical-translation-task.html) with
some of the sources removed for copyright reasons.

============================================================

DATA SUMMARY:
------------

Each corpus has a following format:

    src_sent \t tgt_sent \t data_type \t name_of_original_corpus

The data was shuffled and deduplicated on a sentence-level (each sentence
within a context of two sentences).
The in-domain out-of-domain corpora can be extracted by grepping medical_(corpus|dictionary)
or general_(corpus|dictionary) data_type from the corpus.
The original corpora can be restored by grepping a correct name_of_original_corpus (shuffled).


# Language pair:
# -------------
# num_of_sentences_medical_domain (num_of_sents_after_hard_deduplication)
# num_of_sentences_general_domain (num_of_sents_after_hard_deduplication)

cs-en:
-----
1,145,493 (819,697)
46,949,496(38,065,775)

de-en:
-----
3,036,581 (3,036,581)
34,034,887 (31,638,916)

es-en:
-----
790,915 (631,087)
91,663,568 (75,421,729)

fr-en:
-----
2,812,305 (2,634,229)
84,969,376 (74,045,053)

hu-en:
-----
464,847 (351,336)
48,179,435 (39,499,594)

pl-en:
-----
1,116,773 (800,662)
37,775,703 (31,786,926)

ro-en:
-----
782,682 (852,800)
60,869,087 (47,829,602)

sv-en:
-----
565,028 (444,777)
22,279,720 (19,447,606)

============================================================

UFAL Medical Corpus was supported by:
- EU H2020 project KConnect (contract no. 644753)
- EU H2020 project HimL (contract no. 644402)

============================================================

MAIN CONTRIBUTORS:
-----------------
Ondřej Bojar
Jindřich Libovický
Pavel Pecina
Aleš Tamchyna
Dušan Variš
