# Cover Letter — IOVS Submission

[DATE]

Editor-in-Chief  
Investigative Ophthalmology & Visual Science  
Association for Research in Vision and Ophthalmology

---

Dear Editor,

We submit for your consideration the manuscript entitled **"FA-ICGA-FM: A Comprehensive Deep Learning Pipeline for Multi-Class Classification of Fluorescein Angiography and ICG Angiography Images Across 24 Retinal Conditions"** for publication in *Investigative Ophthalmology & Visual Science*.

**Scientific rationale.** Fluorescein angiography (FA) and ICG angiography (ICGA) are irreplaceable diagnostic tools in clinical retinal practice, yet automated classification across the full spectrum of angiographic findings remains technically challenging. The recent release of the APTOS 2023 dataset—58,000 FA and ICGA images labeled across 24 retinal conditions—provides an unprecedented opportunity for large-scale foundation model benchmarking in this space. No unified, leakage-safe, multi-backbone pipeline for FA/ICGA classification has previously been described.

**What this paper describes.** FA-ICGA-FM is a comprehensive, open-source deep learning pipeline implementing 60 methodological components: nine backbone architectures (ViT-B/16, Swin-B, ConvNeXt-B, RETFound, BiomedCLIP, DINOv2, EfficientNetV2-S, DeiT-III-B, MaxViT-Tiny) plus CLIP zero-shot; self-supervised pre-training via masked autoencoder and cross-modal FA/ICGA contrastive learning (extending the InterpreFFA paradigm); a comprehensive evaluation suite including few-shot, ordinal regression, conformal prediction, calibration, and interpretability tools; and ensemble methods including federated learning simulation.

**Honest disclosure on results.** We must be transparent with the Editors: the pipeline has been validated end-to-end using synthetic smoke-test data, but has not yet been run on the real APTOS 2023 corpus. The APTOS 2023 data requires staging on our institutional HPC cluster before full evaluation can proceed. All numerical results reported in the manuscript derive exclusively from synthetic data (random-label, random-texture images) and are presented solely to confirm pipeline executability, not to characterize real FA/ICGA classification performance. We present this work as a methodology paper describing a validated, reproducible, and open-source infrastructure for FA/ICGA foundation model benchmarking. Real-data results will be reported in a follow-on paper.

**Why IOVS.** IOVS is the leading journal for ophthalmic vision science research, and has published landmark work on deep learning for retinal imaging (including RETFound, the ophthalmic foundation model from which our RETFound backbone is derived). We believe the retinal imaging community would benefit from access to a comprehensive, openly published pipeline for FA/ICGA classification benchmarking, and that this work is well matched to IOVS's readership.

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
