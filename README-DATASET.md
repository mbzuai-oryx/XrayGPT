## High Quality and Interactive Summary Xray Datasets


### Introduction
<hr>

The MIMIC-CXR is a publicly available collection of chest radiographs associated with free-text radiology reports. It consists of 377,110 images and 227,827 associated reports, which are used for both training and testing purposes. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. You can download the raw images from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the raw reports from [here](https://physionet.org/content/mimic-cxr/2.0.0/). The OpenI dataset is a collection of chest x-ray images from the Indiana University hospital network comprises of 6,459 images and 3,955 reports. You can download OpenI dataset and raw reports from [here](https://openi.nlm.nih.gov/faq#collection).

### Preprocessing
<hr>

To generate concise and coherent medical summaries from the unstructured reports, we performed the following pre-processing steps for both datasets:

+ Removal of incomplete reports lacking finding or impression sections.
+ Elimination of reports with finding sections containing less than 10 words.
+ Exclusion of reports with impression sections containing less than 2 words.

In addition, utilizing the power of gpt-3.5-turbo, we implemented the following pre-processing techniques to ensure high-quality summaries per image:

+ Elimination of sentences containing comparisons to the patient's prior medical history.
+ Removal of de-defined symbols "__" while preserving the original meaning.
+ As our training relies on image-text pairs, we excluded the provided view from the summary.
+ We combine the clean findings and impression to generate an interactive and high-quality summary.

Following these steps, we obtained a set of filtered training reports consisting of 114,690 reports associated with 241k training images based on Mimic-CXR dataset. Also, we obtained 3,403 high-quality summaries that used for training based on OpenI dataset.

Here is an example before and after the proposed pre-processing:

**Input findings:**
PA and lateral views of the chest were provided demonstrating no focal consolidation, effusion or pneumothorax. Cardiomediastinal silhouette appears normal and stable.  There is a compression deformity involving a mid thoracic vertebral body, which appears new from the prior chest radiograph of ___.   No free air below the right hemidiaphragm. There are tiny surgical clips in the left base of neck, likely indicating prior thyroid surgery.

**Input Impression:**
No acute intrathoracic process.  Interval development of a mid thoracic spine compression fracture.
 

**High-quality and interactive summary:**
The chest x-ray findings reveal no evidence of focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette appears stable and normal. There is a newly developed mid thoracic spine compression fracture but no free air below the right hemidiaphragm. The presence of surgical clips in the left base of the neck suggests prior thyroid surgery. The impression suggests that there is no acute intrathoracic condition detected in the x-ray aside from the new development of mid thoracic spine compression fracture.
