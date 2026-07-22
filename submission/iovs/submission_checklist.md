# IOVS Submission Checklist — FA-ICGA-FM

**Journal:** Investigative Ophthalmology & Visual Science (IOVS)  
**Target section:** Methods and Instruments  
**Package assembled:** 2026-07-22  
**Status:** READY FOR PI REVIEW (author facts + PI TODOs pending — see below)

---

## Manuscript Requirements

| Requirement | IOVS Spec | This Submission | Status |
|---|---|---|---|
| Word count (body, excl. title/legends/refs) | ≤3,500 | ~3,200 | PASS |
| Structured abstract | ≤250 words; Purpose/Methods/Results/Conclusions | 240 words, 4 sections | PASS |
| Section order | Title → Abstract → Intro → Methods → Results → Discussion → Acks → Refs → Fig Legends → Tables | Followed | PASS |
| Double-spaced, line-numbered | Required | MD source; formatting at submission portal | TODO (portal) |
| Running title | Required | "FA-ICGA-FM: Multi-Backbone Angiography Classification Pipeline" | PASS |
| Numbered references (citation order) | Required | Refs 1–20 numbered sequentially | PASS |
| References: author, title, journal (Index Medicus), year, vol, pages | Required | Applied throughout; arXiv refs marked clearly | PASS |
| >6 authors → "et al." after 3rd | Required | Applied in ref list | PASS |
| Figures cited in numerical order | Required | Fig 1 → 2 → 3 → 4 in text | PASS |

---

## Figures

| Figure | File | Format | DPI | Dimensions | Content | Status |
|---|---|---|---|---|---|---|
| Fig 1 | figures/fig1_pipeline_overview.tiff | TIFF | 300 | 4170×2370 px (13.9×7.9 in) | Pipeline architecture schematic | PASS |
| Fig 2 | figures/fig2_backbone_comparison.tiff | TIFF | 300 | 2968×2070 px (9.9×6.9 in) | Backbone linear-probe AUC bar chart | PASS |
| Fig 3 | figures/fig3_fewshot_calibration.tiff | TIFF | 300 | 3221×1639 px (10.7×5.5 in) | Few-shot + calibration panel | PASS |
| Fig 4 | figures/fig4_methods_coverage.tiff | TIFF | 300 | 2993×2668 px (10.0×8.9 in) | Methods coverage pie chart | PASS |

All figures generated from Output/ data in this repo (real metrics_summary.json). No synthetic values inflated or reported as real. "Synthetic data only" warning on every figure.

---

## Tables

| Table | Content | Status |
|---|---|---|
| Table 1 | Pipeline smoke-test validation summary (all 23 items, status, metric) | In manuscript.md |

---

## Required Statements

| Statement | Required | In Manuscript | PI TODO |
|---|---|---|---|
| Ethics / IRB | Required | Placeholder [IRB-TBD] | PI must add IRB protocol number or state exemption |
| Funding | Required | Placeholder [FUNDING-TBD] | PI must add grant numbers |
| Conflict of Interest | Required | Placeholder [COI-TBD] | PI must confirm/deny COI for all authors |
| Author Contributions (CRediT) | Required | Placeholder [CREDIT-TBD] | PI must assign CRediT roles per author |
| Data Availability | Required | APTOS 2023 URL + pipeline repo URL stated | PASS (repo is private; note for PI: make public before submission or provide access instructions) |
| ORCID | Required for corresponding author; encouraged for all | Placeholder [ORCID-TBD] | PI must add ORCIDs at submission portal |

---

## Author Information

| Field | Status |
|---|---|
| Author list | [AUTHOR-LIST-TBD] — no author lock file found |
| Affiliations | [AFFILIATIONS-TBD] — from corneal_author_facts.json: Dartmouth-Hitchcock / UTHSC |
| Corresponding author | [CORRESPONDING-AUTHOR-TBD] |
| ORCIDs | Only PI ORCID available (0000-0002-0354-0569 for Benton Chuter); others TBD |

**From corneal_author_facts.json (for reference):**
- PI: Benton Chuter, MD, MS — ORCID 0000-0002-0354-0569 — Benton.Chuter@hitchcock.org
- Co-authors listed: Fae B. Kayarian, Bailey R. Millis, Monica M. Jablonski PhD, William G. Gensheimer MD
- Affiliations: DHMC / Dartmouth-Hitchcock; UTHSC Hamilton Eye Institute; Geisel School of Medicine

---

## Artifacts Assembled

```
submission/iovs/
├── manuscript.md              ← full manuscript (Title → References → Fig Legends → Table 1)
├── cover_letter.md            ← cover letter draft (PI TODO: add reviewers, sign)
├── submission_checklist.md    ← this file
├── generate_figures.py        ← reproducible figure generation script
└── figures/
    ├── fig1_pipeline_overview.tiff
    ├── fig2_backbone_comparison.tiff
    ├── fig3_fewshot_calibration.tiff
    └── fig4_methods_coverage.tiff
```

---

## PI Action Items Before Submission

1. **Author lock** — Run author-lock step in web console to lock byline, affiliations, CRediT, ORCIDs for all 5 co-authors. When locked, re-run step 4 to fill placeholders.
2. **IRB** — Add IRB protocol number (or exemption statement) to manuscript ethics section.
3. **Funding** — Add grant/funding acknowledgment to manuscript and cover letter.
4. **COI declaration** — Confirm no COI for all authors (or disclose).
5. **Pipeline repo visibility** — The pipeline repo (FA-ICGA-FM-Pipeline) is currently PRIVATE. Change to PUBLIC before submission, or add a data availability note that it will be made public upon acceptance.
6. **Suggested reviewers** — Add 3–5 suggested reviewers to cover letter.
7. **APTOS 2023 data staging** — Stage the APTOS 2023 dataset at `/dartfs-hpc/scratch/f008pp2/data/aptos2023/` to enable real-data validation; manuscript should be updated with real results before final submission.
8. **Journal decision** — IOVS may require real results for a Methods paper. If reviewers require real-data performance numbers, the authors should run the full pipeline after APTOS 2023 staging and submit an updated version. Alternatively, consider framing as a "Technical Report" or "Software Report" if IOVS offers this article type.
9. **APC** — Confirm APC payment arrangement with IOVS (open access required or optional).
10. **Submission portal** — Format manuscript as double-spaced, line-numbered DOCX at the IOVS submission portal (https://iovs.msubmit.net/cgi-bin/main.plex). Convert manuscript.md → DOCX for upload.

---

## BNL Verdict Reference

- **Verdict:** descriptive (SMOKE-ONLY; synthetic data)
- **Best AUC (synthetic):** 0.784 (ViT-B/16 linear probe)
- **60 IMPLEMENT-DONE items** across BNL-1 through BNL-4
- **Dual-audit:** PASS (leakage-safe; no AI attribution; no fabricated numbers)
- **Pipeline repo:** Chuter-lab/FA-ICGA-FM-Pipeline (HEAD cb9da61)
- **Manuscript repo:** Chuter-lab/FA-ICGA-FM (HEAD ac3acff + this submission commit)
