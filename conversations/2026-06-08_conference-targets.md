# Conference targets for a thesis-based paper

Date checked: 2026-06-08.

Question: find nearby conferences where a paper based on the thesis on statistical inference for linear stochastic approximation with Markovian noise, Polyak--Ruppert averaging, and Richardson--Romberg extrapolation could be submitted.

## Short recommendation

The best near-term target, if the thesis can be compressed into a 6-page control-style paper, is ACC 2027. The best ML/statistics target with an actual 2026 deadline is AAAI 2027, but only if the paper is framed as theory for stochastic approximation in AI/RL/optimization and the main theorem is clean enough for a broad AI audience. FSML 2026 is attractive as a non-archival workshop submission, especially if the goal is feedback and visibility rather than archival publication.

For a more natural archival ML/statistics paper, the realistic cycle is probably AISTATS 2027 / ALT 2027 / COLT 2027 / UAI 2027 / ICML 2027. Their 2027 CFPs were not all officially posted when checked, but the 2026 schedules show the expected windows: AISTATS and ALT around September--October, ICML around January, and UAI/COLT around February.

## Reputation ranking

Among the listed venues, the most prestigious broad ML/theory targets are:

1. COLT / ICML / AISTATS. COLT is the most theory-selective; ICML is the most visible broad ML venue; AISTATS is probably the best statistical-ML match for an inference theorem.
2. UAI / ALT / AAAI. UAI is strong for uncertainty and probabilistic/statistical ML; ALT is specialized and respected in learning theory; AAAI is broad, visible, and selective, but less naturally aligned with a proof-heavy stochastic-approximation paper unless the AI framing is strong.
3. ACC. Very respectable in control and systems, and probably the best reputable near-term fit for stochastic approximation / Markovian linear recursions. It is not as prestigious as COLT/ICML/AISTATS from an ML-theory perspective, but it is a serious archival venue.
4. FSML workshop. Scientifically relevant and useful for feedback, but non-archival and less career-legible than a main conference paper.
5. ICSMD / ICMSA / ECSO-CMS. Potentially fine for presentation or backup, but much less prestigious for this thesis topic than the venues above.
6. AIMLSystems. Respectable as a systems-oriented venue, but the fit is weak for a mostly mathematical thesis unless the paper is rewritten around systems or experiments.

Practical interpretation: if the goal is maximum prestige, aim for AISTATS/ALT/COLT/UAI/ICML after polishing. If the goal is a reputable and realistic first archival submission on the current timeline, ACC 2027 is the best target. If the goal is early feedback without consuming the archival result, FSML 2026 is useful.

## CORE / ICORE rank breakdown

Here "A*" means the CORE/ICORE computer-science conference ranking. This is useful for CS/ML venues, but it is not a universal ranking for statistics, control, or workshops.

| Venue | Current CORE/ICORE rank checked | Comment |
| --- | --- | --- |
| ICML | A* | Top broad ML venue. |
| COLT | A* | Top learning-theory venue. |
| AAAI | A* | Broad AI A* venue, but topical fit depends on AI/RL framing. |
| AISTATS | A | Strong statistical ML fit, but not A* in ICORE2026. |
| UAI | A | Was A* in older CORE lists; ICORE2026 lists it as A. |
| ALT | B | Specialized and respected in learning theory, but ICORE2026 lists it as B. |
| ACC | No current useful CORE/ICORE rank found | CORE is CS-focused; control venues are not well represented. Do not interpret this as low control-community reputation. |
| FSML workshop | Not ranked / workshop | Non-archival workshop; CORE rank is not the right metric. |
| AIMLSystems | Not found in checked CORE/ICORE results | Possibly too new or not ranked in the database. |
| ICSMD / ICMSA / ECSO-CMS | Not found in checked CORE/ICORE results | More presentation/backup venues for this thesis topic. |

If the institution or grant explicitly cares about CORE A*, the clean A* choices from the current list are ICML, COLT, and AAAI. If the scientific fit matters more than the literal A* label, AISTATS remains one of the best targets for this thesis.

## Open or near-open options

### AIMLSystems 2026

- Deadline: 2026-06-22, 11:59 AoE, after an extension.
- Conference: 2026-10-06 to 2026-10-09, Lecco, Italy.
- Source: https://www.aimlsystems.org/2026/
- Fit: weak-to-medium. Useful only if the thesis paper is written around scalable/efficient AI-ML systems or computational diagnostics for stochastic approximation. A proof-heavy RR/CLT paper is not a natural fit.
- My take: do not prioritize unless there is already a strong experimental systems story.

### AAAI 2027

- Abstract deadline: 2026-07-21, 11:59 PM UTC-12.
- Full paper deadline: 2026-07-28, 11:59 PM UTC-12.
- Conference: 2027-02-16 to 2027-02-23, Montreal, Canada.
- Source: https://aaai.org/conference/aaai/aaai-27/
- Fit: medium. The venue covers AI broadly; the paper would need a clear AI/RL/optimization motivation, not only stochastic-process asymptotics. A good framing is: confidence intervals for constant-stepsize stochastic approximation with Markovian data, with RR reducing stepsize bias.
- Risk: high selectivity and broad reviewing. If the current theorem still has proof gaps, this is too soon.

### ICSMD 2026

- Deadline: 2026-08-20.
- Conference: 2026-11-13 to 2026-11-15, Chengdu, China.
- Source: https://ieee-ims.org/event/conference/2026-international-conference-sensing-measurement-data-analytics-era-artificial
- Fit: weak-to-medium. It is an IEEE instrumentation/measurement/data analytics conference, so the paper would need an applied data-analytics or measurement-inference angle.
- My take: backup venue, not the natural first choice for the theory.

### FSML 2026, Frontiers in Statistical Machine Learning

- Deadline: 2026-08-31 AoE.
- Workshop: 2026-12-14, Split, Croatia, co-located with ICSDS 2026.
- Format: non-archival 3--5 page extended abstract for new/work-in-progress research.
- Source: https://fsml-ims-workshop.org/
- Fit: medium-to-good for feedback if the paper is positioned as statistical ML theory. Since it is non-archival, it should not block a later AISTATS/ALT/COLT/UAI submission.
- My take: good low-risk place to present the thesis idea while preparing a full archival version.

### ICMSA 2026

- Full-paper deadline: 2026-08-31.
- Abstract / contributed talk / poster submissions: through 2026-10-31.
- Conference: 2026-12-09 to 2026-12-11, Chiang Mai, Thailand.
- Source: https://iasc-ars2026.icdi.cmu.ac.th/ICMSA_IMT
- Fit: medium as a regional mathematics/statistics venue. It explicitly lists statistical learning theory, applied statistics, probability, uncertainty quantification, and optimization.
- Caveat: the conference proceedings option is stated as not indexed in Scopus/Web of Science; partner journal options are separate and competitive.
- My take: acceptable backup or presentation venue, not the main prestige target.

### ACC 2027

- Joint ACC + L-CSS manuscript deadline: 2026-09-11.
- ACC manuscript deadline: 2026-09-25.
- Conference: 2027-07-07 to 2027-07-09, Philadelphia, USA.
- Source: https://acc2027.a2c2.org/
- CFP source: https://acc2027.a2c2.org/wp-content/uploads/2025/10/ACC_2027_CFP-2025-10-17.pdf
- Fit: good if the paper is written as stochastic approximation / adaptive control / inference for linear recursions under Markovian noise. ACC regular papers are short, so the paper must focus on one clean result, not the whole thesis.
- Suggested angle: "Richardson--Romberg bias reduction for Polyak--Ruppert averaged constant-stepsize linear stochastic approximation under Markovian noise."
- My take: strongest concrete near-term archival target.

### Stochastic Systems special issue, DaiFest 2026

- Deadline: 2026-10-15.
- Source: https://pubsonline.informs.org/page/stsy/calls-for-papers
- Fit: good thematically for stochastic systems / applied probability / reinforcement learning for stochastic systems, but this is a journal special issue rather than a conference.
- My take: consider if the thesis becomes a full journal-style manuscript and the Markovian stochastic-system contribution is emphasized.

### ECSO-CMS 2027

- Abstract and best student paper submissions: official page says they open in fall 2026.
- Conference: 2027-06-20 to 2027-06-23, Lancaster, UK.
- Source: https://wp.lancs.ac.uk/ecso-cms-2027/
- Fit: medium. Good for stochastic optimization / decision-making under uncertainty, less direct for statistical inference unless connected to stochastic approximation algorithms.
- My take: put on the watchlist, especially if there is a best-student-paper track.

## Watchlist for the natural ML/stat theory cycle

These are likely better scientific fits, but the 2027 calls should be checked again when officially posted.

### AISTATS 2027

- 2027 CFP not found as official at the time checked.
- 2026 reference: abstract deadline 2025-09-25, full paper deadline 2025-10-02.
- Source for 2026 CFP: https://virtual.aistats.org/Conferences/2026/CallForPapers
- Fit: very good if the paper states a statistical inference theorem, a CLT/Berry--Esseen bound, or confidence interval result for stochastic approximation.
- Preparation target: have an 8-page main paper by early September 2026.

### ALT 2027

- 2027 CFP not found as official at the time checked.
- 2026 reference: paper deadline 2025-10-02; conference dedicated to theoretical and algorithmic aspects of ML.
- Source for 2026 CFP: https://algorithmiclearningtheory.org/alt2026/call-for-papers/
- Fit: good for a clean theory paper, especially if the result is about stochastic approximation, online/stochastic optimization, or RL.
- Preparation target: proof-complete paper by September 2026.

### ICML 2027

- 2027 CFP not found as official at the time checked.
- 2026 reference: abstract deadline 2026-01-23, full paper deadline 2026-01-28.
- Source for 2026 CFP: https://icml.cc/Conferences/2026/CallForPapers
- Fit: good but ambitious. ICML requires the contribution to be clearly significant for ML, not only for asymptotic stochastic-process theory.
- Preparation target: mature empirical section and crisp theorem by December 2026.

### UAI 2027

- 2027 CFP not found as official at the time checked.
- 2026 reference: paper deadline 2026-02-25; accepted papers in PMLR.
- Source for 2026 CFP: https://www.auai.org/uai2026/call_for_papers
- Fit: good if the thesis is framed around uncertainty quantification and inference for stochastic approximation.
- Preparation target: submit if the confidence-interval story is central and readable.

### COLT 2027

- 2027 CFP not found as official at the time checked.
- 2026 reference: deadline 2026-02-04; accepted papers in PMLR.
- Source for 2026 CFP: https://learningtheory.org/colt2026/cfp.html
- Fit: high bar. COLT is natural only if the paper has a theorem that is both novel and cleanly comparable to learning-theory literature.
- Preparation target: only pursue if the main theorem is proof-complete and the novelty over Huo et al., Samsonov et al., and Levin et al. is unmistakable.

## Practical submission strategy

1. If a short archival paper is possible by September: aim for ACC 2027.
2. If the paper is more statistical-ML than control: prepare for AISTATS 2027 / ALT 2027 around September--October 2026, checking the official calls in August.
3. Submit a 3--5 page non-archival version to FSML 2026 if feedback and visibility are useful.
4. Use AAAI 2027 only if the paper can be reframed for a broad AI audience by late July.
5. Keep UAI/COLT/ICML 2027 for the polished version after the diploma proof gaps are closed.
