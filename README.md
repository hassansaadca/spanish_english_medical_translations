# Spanish to English Translations in a Medical Context

There are many people in the United States for whom English is a second language, which may hinder their ability to get access to essential services like medical care. In addition to difficulties inherent in interpreting everyday language, industry-specific terminology adds another degree of complexity between native English and native Spanish speakers. 

While medical interpreters do exist to assist with these tasks, they are usually short staffed, and their time is best used in live (active) translation scenarios rather than document-based (passive) translation tasks. Our model will allow for the full translation from Spanish to English in the  written form to assist with the passive element. We envision a tool that will allow Spanish-speaking patients or family members to type in a phrase which can then be interpreted to communicate with a medical staff member. 

## Model
We address the process of building a model to translate Spanish to English in the biomedical domain. We start with an existing model sourced from HuggingFace and we train on a text corpus from the Institute of Formal and Applied Linguistics (UFAL). We report the evaluation metrics both in terms of the corpus BLEU scores and a comparison to translations completed by a professional Spanish-English medical interpreter.

![Model Architecture](/images/model_architecture.png)

The MarianMT transformer consists of an encoder-decoder stack with 6 layers in each component. Similar to T5, it uses a self attention layer on the encoder side and a masked self-attention layer on the decoder side to act as an autoregressive language model when converting from embeddings to text. Where T5 is trained on various different tasks including question-answering, translation, etc., our fine-tuning of course only focused on a translation-based training strategy. 

![Model Preprocessing](/images/model_preprocessing.png)

## Data
The [UFAL](https://github.com/ufal) corpus provides access to a 13.1 GB dataset of text pairs between English and several other languages (e.g. French, Spanish, Hungarian,  Romanian, etc.). Over 430M text pairs exist across these languages, and roughly 10M of them consist of text that was extracted from medical documents.
