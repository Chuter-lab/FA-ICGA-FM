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

**Results status.** All reported values derive from end-to-end validation on synthetic smoke-test data; the APTOS 2023 corpus awaits staging on our institutional HPC cluster. The validated infrastructure itself—leakage-safe split logic, nine backbone architectures, 60 implemented methods, confirmed end-to-end executability—constitutes the primary scientific contribution. Benchmark platform papers are established in the ophthalmic imaging literature, and open-source release at this stage enables independent groups to audit, extend, and reproduce the benchmark as soon as the APTOS 2023 data is accessible. Real-data performance results will be reported in a follow-on submission.

**Why IOVS.** IOVS is the leading venue for ophthalmic vision science and routinely publishes methods advances in retinal imaging. FA and ICGA are central to retinal diagnosis and are heavily represented in IOVS's clinical and translational content; the retinal imaging community that interprets these studies is precisely the audience for a standardized benchmarking platform. Practitioners and translational researchers evaluating deep learning for FA/ICGA will benefit most from access to an openly documented, leakage-safe pipeline, and this work is well matched to IOVS's scope and readership.

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
