# Cover Letter — IOVS Submission

[DATE]

Editor-in-Chief  
Investigative Ophthalmology & Visual Science  
Association for Research in Vision and Ophthalmology

---

Dear Editor,

We submit for your consideration the manuscript entitled **"FA-ICGA-FM: A Multi-Backbone, Leakage-Safe Deep Learning Pipeline for Fluorescein and ICG Angiography Classification Across 24 Retinal Conditions"** for publication in *Investigative Ophthalmology & Visual Science*.

**Scientific rationale.** Fluorescein angiography (FA) and ICG angiography (ICGA) are irreplaceable diagnostic tools in clinical retinal practice, yet automated classification across the full spectrum of angiographic findings remains technically challenging. The recent release of the APTOS 2023 dataset—58,000 FA and ICGA images labeled across 24 retinal conditions—provides an unprecedented opportunity for large-scale foundation model benchmarking in this space. No unified, leakage-safe, multi-backbone pipeline for FA/ICGA classification has previously been described, and comparative studies in adjacent ophthalmic imaging domains have shown that apparent performance advantages between architectures frequently fail to generalize when evaluated independently—making a shared, standardized benchmark platform a prerequisite for meaningful progress.

**What this paper describes.** FA-ICGA-FM is an open-source deep learning pipeline implementing 60 methodological components: nine backbone architectures (ViT-B/16, Swin-B, ConvNeXt-B, RETFound, BiomedCLIP, DINOv2, EfficientNetV2-S, DeiT-III-B, MaxViT-Tiny) plus CLIP zero-shot; self-supervised pre-training via masked autoencoder and cross-modal FA/ICGA contrastive learning (extending the InterpreFFA paradigm); a multi-protocol evaluation suite including few-shot, ordinal regression, conformal prediction, calibration, and interpretability tools; and ensemble methods including federated learning simulation.

**Results status.** All numerical results in this submission derive from end-to-end validation on synthetic smoke-test data; the APTOS 2023 corpus awaits staging on our institutional HPC cluster. The pipeline is submitted as a methods paper describing validated, reproducible, open-source infrastructure for FA/ICGA foundation model benchmarking. Real-data performance results will be reported in a follow-on paper once the corpus is staged.

**Why IOVS.** IOVS is the leading journal for ophthalmic vision science research and has published landmark work on deep learning for retinal imaging, including RETFound, the ophthalmic-specific retinal foundation model that is one of the nine backbones benchmarked here. The retinal imaging community will benefit from access to an openly published pipeline for FA/ICGA classification benchmarking, and this work is well matched to IOVS's readership.

**Prior presentation.** This work has not been presented or published elsewhere.

**Suggested reviewers.** [SUGGESTED-REVIEWERS-TBD — PI to complete]

**Conflicts of interest:** [COI-TBD]

**Funding:** [FUNDING-TBD]

We thank you for considering this manuscript and look forward to your response.

Sincerely,

[CORRESPONDING-AUTHOR-TBD], [DEGREE-TBD]  
[INSTITUTION-TBD]  
[EMAIL-TBD]  
[PHONE-TBD]
