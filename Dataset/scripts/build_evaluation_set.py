"""
Build a 30-item evaluation/ablation QA dataset for comparing VectorRAG vs
GraphRAG vs HybridRAG on the Alzheimer's corpus. Every question is grounded
in the actual fetched sources (see 'Selected articles/') and, where relevant,
in the entity/relation graph (see 'Knowledge graph/'). Answers are ready to
use directly as gold references (not flagged for review, per user request).

Question design:
  - 10 single-hop fact-lookup questions -> answerable from one chunk/article,
    should favor VectorRAG (designed_to_favor: "vectorrag")
  - 10 multi-hop / relational questions -> require connecting 2+ entities or
    articles not co-located in a single chunk, should favor GraphRAG
    (designed_to_favor: "graphrag")
  - 10 broad synthesis / comparison questions -> require both wide corpus
    coverage and relational reasoning, should favor HybridRAG
    (designed_to_favor: "hybridrag")
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent.parent / "Evaluation"
OUT_DIR.mkdir(exist_ok=True)

QA_PAIRS = [
    # ---- vectorrag: single-hop fact lookup ----
    {
        "question_id": "q01",
        "question": "What is the strongest genetic risk factor for sporadic Alzheimer's disease?",
        "type": "fact_lookup",
        "expected_answer": "The APOE epsilon-4 (e4) allele is the strongest genetic risk factor for sporadic Alzheimer's disease, confirmed by large-scale GWAS meta-analyses.",
        "supporting_sources": ["pubmed_33340485"],
        "reasoning_path": ["APOE_e4", "AD"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q02",
        "question": "What is the strongest genetic protective factor against Alzheimer's disease?",
        "type": "fact_lookup",
        "expected_answer": "The APOE epsilon-2 (e2) allele is the strongest known genetic protective factor against Alzheimer's disease.",
        "supporting_sources": ["pubmed_33340485"],
        "reasoning_path": ["APOE_e2", "AD"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q03",
        "question": "Which three genes, when mutated, cause the familial (autosomal-dominant) form of Alzheimer's disease?",
        "type": "fact_lookup",
        "expected_answer": "APP (amyloid precursor protein), PSEN1 (presenilin 1), and PSEN2 (presenilin 2).",
        "supporting_sources": ["pubmed_23276979", "textbook_statpearls_alzheimer_disease"],
        "reasoning_path": ["APP", "PSEN1", "PSEN2", "AD"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q04",
        "question": "Roughly what percentage of dementia cases worldwide is caused by Alzheimer's disease?",
        "type": "fact_lookup",
        "expected_answer": "Alzheimer's disease accounts for approximately 60-70% of dementia cases worldwide.",
        "supporting_sources": ["wikipedia_dementia"],
        "reasoning_path": ["AD", "Dementia"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q05",
        "question": "When did lecanemab receive full (traditional) FDA approval, as opposed to accelerated approval?",
        "type": "fact_lookup",
        "expected_answer": "Lecanemab received accelerated FDA approval in January 2023, then was converted to full traditional approval in July 2023.",
        "supporting_sources": ["wikipedia_lecanemab"],
        "reasoning_path": ["Lecanemab", "AD"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q06",
        "question": "What was the first cholinesterase inhibitor licensed for Alzheimer's treatment, and why is it no longer used?",
        "type": "fact_lookup",
        "expected_answer": "Tacrine was the first licensed cholinesterase inhibitor for Alzheimer's disease; it is no longer used because of its hepatotoxicity.",
        "supporting_sources": ["pubmed_33035532"],
        "reasoning_path": ["Tacrine", "CholinesteraseInhibitors", "Hepatotoxicity"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q07",
        "question": "What is memantine's mechanism of action in treating Alzheimer's disease?",
        "type": "fact_lookup",
        "expected_answer": "Memantine is an uncompetitive NMDA (N-methyl-D-aspartate) receptor antagonist that reduces glutamate-driven excitotoxicity.",
        "supporting_sources": ["pubmed_12672860", "wikipedia_memantine"],
        "reasoning_path": ["Memantine", "NMDAReceptor"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q08",
        "question": "What serious adverse effect is associated with anti-amyloid monoclonal antibody treatments like lecanemab?",
        "type": "fact_lookup",
        "expected_answer": "ARIA (amyloid-related imaging abnormalities) -- brain edema and/or microhemorrhage -- is the key adverse effect requiring MRI monitoring.",
        "supporting_sources": ["wikipedia_lecanemab", "pubmed_37955845"],
        "reasoning_path": ["Lecanemab", "ARIA"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q09",
        "question": "Approximately how much did anti-amyloid antibody trials slow cognitive decline?",
        "type": "fact_lookup",
        "expected_answer": "Approximately 30% slowing of cognitive decline was observed in successful anti-amyloid antibody trials.",
        "supporting_sources": ["pubmed_37955845"],
        "reasoning_path": ["AntiAmyloidAntibodies", "CognitiveDecline"],
        "difficulty": "easy",
        "designed_to_favor": "vectorrag",
    },
    {
        "question_id": "q10",
        "question": "What genetic condition is considered strong evidence for the amyloid cascade hypothesis because it causes an extra copy of the APP gene?",
        "type": "fact_lookup",
        "expected_answer": "Down syndrome (trisomy 21) -- the extra chromosome 21 copy means an extra APP gene copy, supporting the amyloid hypothesis.",
        "supporting_sources": ["textbook_statpearls_alzheimer_disease"],
        "reasoning_path": ["DownSyndrome", "Chr21", "APP", "AmyloidCascadeHypothesis"],
        "difficulty": "medium",
        "designed_to_favor": "vectorrag",
    },
    # ---- graphrag: multi-hop / relational ----
    {
        "question_id": "q11",
        "question": "How does carrying the APOE e4 allele interact with sex to affect Alzheimer's risk?",
        "type": "multi_hop",
        "expected_answer": "APOE e4 is the strongest genetic risk allele for AD, but its risk-raising effect is not sex-neutral: carrying APOE4 increases AD risk more in women than in men, consistent with broader findings that sex/gender modify AD risk-factor impact and cognitive resilience.",
        "supporting_sources": ["pubmed_33340485", "wikipedia_apolipoprotein_e", "pubmed_38967222"],
        "reasoning_path": ["APOE_e4", "AD", "SexDifferences"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q12",
        "question": "Trace the chain of events by which poor sleep is thought to contribute to dementia-related protein aggregation.",
        "type": "multi_hop",
        "expected_answer": "Sleep disturbance impairs the glymphatic system, the brain's waste-clearance pathway that is mostly active during sleep and which normally helps clear amyloid-beta and other aggregation-prone proteins; the glymphatic system also degrades with age, so impaired sleep plus aging compounds reduced clearance, contributing to protein aggregation and dementia onset.",
        "supporting_sources": ["pubmed_33004510"],
        "reasoning_path": ["SleepDisturbance", "GlymphaticSystem", "AmyloidBeta", "Aging"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q13",
        "question": "How do microglia connect the amyloid hypothesis and the neuroinflammation hypothesis of Alzheimer's disease?",
        "type": "multi_hop",
        "expected_answer": "Microglia are the primary mediators of neuroinflammation in AD, and they directly interact with both amyloid-beta and tau pathology -- meaning microglial activity sits at the intersection of the amyloid-centric and neuroinflammation-centric explanations of disease progression, and evidence for the latter has weakened purely amyloid-centric models.",
        "supporting_sources": ["pubmed_36911732"],
        "reasoning_path": ["Microglia", "AmyloidBeta", "TauProtein", "Neuroinflammation", "AmyloidCascadeHypothesis"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q14",
        "question": "Why is Down syndrome considered a natural model supporting the amyloid cascade hypothesis, tracing the full genetic-to-molecular chain?",
        "type": "multi_hop",
        "expected_answer": "Down syndrome is caused by trisomy of chromosome 21; since the APP gene sits on chromosome 21, individuals with Down syndrome carry an extra APP gene copy, leading to greater APP expression and amyloid-beta production -- which is exactly the mechanism the amyloid cascade hypothesis proposes drives Alzheimer's pathology.",
        "supporting_sources": ["textbook_statpearls_alzheimer_disease"],
        "reasoning_path": ["DownSyndrome", "Chr21", "APP", "AmyloidBeta", "AmyloidCascadeHypothesis"],
        "difficulty": "hard",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q15",
        "question": "How are vascular risk factors, vascular dementia, and mixed dementia related to Alzheimer's disease?",
        "type": "multi_hop",
        "expected_answer": "Cardiovascular/vascular risk factors raise the risk of vascular dementia; vascular dementia frequently co-occurs with Alzheimer's disease as 'mixed dementia,' which is in fact the most common mixed-dementia presentation -- so managing vascular risk is relevant to both pure vascular dementia and AD-vascular comorbidity.",
        "supporting_sources": ["pubmed_39889875", "wikipedia_dementia"],
        "reasoning_path": ["CardiovascularRisk", "VascularDementia", "MixedDementia", "AD"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q16",
        "question": "Describe the three distinct biological mechanisms through which APOE genotype is now thought to influence Alzheimer's pathology, beyond its original amyloid-only role.",
        "type": "multi_hop",
        "expected_answer": "Beyond its originally described role in amyloid-beta aggregation and clearance, APOE genotype is now linked to (1) tau neurofibrillary degeneration, (2) microglia and astrocyte inflammatory responses, and (3) blood-brain barrier disruption.",
        "supporting_sources": ["pubmed_33340485"],
        "reasoning_path": ["APOE", "AmyloidBeta", "TauProtein", "Microglia"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q17",
        "question": "How does mild cognitive impairment connect to Alzheimer's disease and the biomarkers used to track its progression?",
        "type": "multi_hop",
        "expected_answer": "Amnestic MCI is frequently a prodromal stage of Alzheimer's disease, converting to probable AD at roughly 10-15% per year; a low plasma Abeta42/40 ratio can strongly support an AD diagnosis even at the MCI/early-dementia stage, and structural MRI can track progressive gray-matter loss as MCI converts toward full AD dementia.",
        "supporting_sources": ["wikipedia_mild_cognitive_impairment", "pubmed_38849944"],
        "reasoning_path": ["MCI", "AD", "FluidBiomarkers", "MRIAtrophy"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q18",
        "question": "Compare how cholinesterase inhibitors perform in MCI versus established Alzheimer's disease.",
        "type": "multi_hop",
        "expected_answer": "In established AD, cholinesterase inhibitors (donepezil, rivastigmine, galantamine) are the main symptomatic drug class, with roughly similar and mild efficacy. In MCI specifically, results are weaker: donepezil showed only minor, short-term benefit with notable side effects, and rivastigmine failed to stop or slow progression to Alzheimer's disease at all.",
        "supporting_sources": ["pubmed_33035532", "wikipedia_mild_cognitive_impairment"],
        "reasoning_path": ["CholinesteraseInhibitors", "Donepezil", "Rivastigmine", "MCI", "AD"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q19",
        "question": "What connects physical activity, air pollution, and vascular risk factors as a category of Alzheimer's risk modifiers?",
        "type": "multi_hop",
        "expected_answer": "All three are modifiable, non-genetic environmental/lifestyle factors that epidemiological studies have linked to Alzheimer's risk: physical activity is protective (lower AD and all-cause dementia incidence), while air pollution (fine particulate matter) and vascular/cardiovascular risk factors both raise AD and dementia risk -- making all three candidate targets for AD prevention strategies.",
        "supporting_sources": ["pubmed_34101789", "pubmed_35301183", "pubmed_39889875"],
        "reasoning_path": ["PhysicalActivity", "AirPollution", "CardiovascularRisk", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "graphrag",
    },
    {
        "question_id": "q20",
        "question": "How does the economic burden of Alzheimer's disease relate to caregiving arrangements?",
        "type": "multi_hop",
        "expected_answer": "Indirect caregiving cost (lost productivity, unpaid family care) is the main contributor to AD's overall societal cost. When formal caregiving accommodation (e.g. a care facility) is used instead, costs shift from indirect cost to direct nonmedical cost, and formal accommodation can raise direct costs to as much as 67.3% of the overall economic burden.",
        "supporting_sources": ["pubmed_37972428"],
        "reasoning_path": ["CaregiverBurden", "EconomicBurden", "AD"],
        "difficulty": "medium",
        "designed_to_favor": "graphrag",
    },
    # ---- hybridrag: broad synthesis / comparison ----
    {
        "question_id": "q21",
        "question": "Compare the mechanisms, treatment goals, and clinical outcomes of cholinesterase inhibitors, memantine, and anti-amyloid monoclonal antibodies for Alzheimer's disease.",
        "type": "synthesis",
        "expected_answer": "Cholinesterase inhibitors (donepezil, rivastigmine, galantamine) are purely symptomatic: they raise synaptic acetylcholine and produce mild, possibly non-clinically-significant benefit. Memantine is also symptomatic, blocking NMDA receptor overstimulation by glutamate, and reduced clinical deterioration in moderate-to-severe AD in trials. Anti-amyloid monoclonal antibodies (lecanemab, aducanumab) are disease-modifying: they directly clear brain amyloid plaque and produced roughly 30% slowing of cognitive decline in trials, but carry a distinct risk -- ARIA (brain edema/microhemorrhage) -- not seen with the older symptomatic drug classes.",
        "supporting_sources": ["pubmed_33035532", "pubmed_12672860", "pubmed_37955845", "wikipedia_lecanemab"],
        "reasoning_path": ["CholinesteraseInhibitors", "Memantine", "NMDAReceptor", "AntiAmyloidAntibodies", "ARIA", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q22",
        "question": "How do the amyloid cascade hypothesis and the neuroinflammation hypothesis of Alzheimer's disease both explain the disease and compete with each other?",
        "type": "synthesis",
        "expected_answer": "The amyloid cascade hypothesis holds that amyloid-beta accumulation is the central driver of AD, initiating downstream tau pathology, synaptic loss, and neurodegeneration -- consistent with amyloid plaques and tau tangles being AD's two hallmark pathologies. However, evidence of elevated inflammatory markers and innate-immune AD risk genes suggests this hypothesis is incomplete: microglia-mediated neuroinflammation interacts independently with both amyloid-beta and tau, and researchers increasingly see disease-promoting and protective factors interacting with, rather than being strictly downstream of, the core amyloid mechanism.",
        "supporting_sources": ["pubmed_36911732", "pubmed_30135715", "pubmed_26921134"],
        "reasoning_path": ["AmyloidCascadeHypothesis", "AmyloidBeta", "NFT", "Neuroinflammation", "Microglia"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q23",
        "question": "Synthesize how genetic factors (APOE, PSEN1/2, APP) and modifiable lifestyle/environmental factors together shape an individual's overall Alzheimer's risk profile.",
        "type": "synthesis",
        "expected_answer": "Genetic risk operates on two tiers: rare autosomal-dominant mutations in APP, PSEN1, or PSEN2 cause early-onset familial AD with near-complete penetrance, while common APOE allele variation (e4 raises risk, e2 lowers it) modulates risk in the much more common sporadic form. Layered on top of this genetic baseline, modifiable factors meaningfully shift risk in either direction: physical activity and cardiovascular health are protective, while vascular risk factors, air pollution, and poor sleep (via impaired glymphatic clearance) raise risk. AD is explicitly described as a multifactorial disorder determined by the interaction of genetic susceptibility and environmental factors across the life course, with early-life and midlife interventions offering the most potential to shift outcomes.",
        "supporting_sources": ["pubmed_23276979", "pubmed_33340485", "pubmed_34101789", "pubmed_35301183", "pubmed_39889875", "pubmed_33004510"],
        "reasoning_path": ["APP", "PSEN1", "PSEN2", "APOE_e4", "APOE_e2", "PhysicalActivity", "CardiovascularRisk", "AirPollution", "SleepDisturbance", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q24",
        "question": "Trace the full biological cascade from an APP gene mutation to the synaptic dysfunction and cognitive decline seen clinically in Alzheimer's disease.",
        "type": "synthesis",
        "expected_answer": "An APP gene mutation alters how amyloid precursor protein is cleaved, increasing production of the amyloid-beta fragment. Amyloid-beta aggregates into extracellular plaques and, together with hyperphosphorylated tau (which aggregates into intracellular neurofibrillary tangles), causes reduction in synaptic strength and synaptic loss. This synaptic dysfunction and downstream neurodegeneration is what produces the clinically measurable progressive decline in cognition and function that defines Alzheimer's disease.",
        "supporting_sources": ["pubmed_23276979", "pubmed_31753135", "pubmed_33986301"],
        "reasoning_path": ["APP", "AmyloidBeta", "AmyloidPlaques", "TauProtein", "NFT", "SynapticDysfunction", "CognitiveDecline", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q25",
        "question": "How do CSF biomarkers, PET imaging, and structural MRI collectively support diagnosing and staging Alzheimer's disease from its preclinical to symptomatic stages?",
        "type": "synthesis",
        "expected_answer": "CSF amyloid-beta42 and tau are established core fluid biomarkers reflecting the disease's molecular hallmarks. Amyloid PET visualizes brain plaque burden in vivo and is increasingly used clinically, while tau PET offers additional value for differential diagnosis and clinical trial patient selection. Structural MRI complements these by tracking hippocampal and cortical atrophy, an established outcome marker of neurodegeneration. Together, these modalities let researchers define 'preclinical Alzheimer's disease' -- biomarker evidence of AD pathology in cognitively healthy individuals -- extending diagnosis earlier than clinical symptoms alone would allow, which is also reflected in the StatPearls textbook's staged clinical/pathological description of the disease.",
        "supporting_sources": ["pubmed_34456336", "pubmed_26921134", "pubmed_26827786", "textbook_statpearls_alzheimer_disease"],
        "reasoning_path": ["FluidBiomarkers", "AmyloidPET", "TauPET", "MRIAtrophy", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q26",
        "question": "How do sex differences and APOE genotype jointly shape resilience to cognitive decline in aging and Alzheimer's disease?",
        "type": "synthesis",
        "expected_answer": "Sex and gender significantly impact the prevalence of protective and risk factors and influence the burden of both AD pathology (amyloid and tau) and comorbid pathologies like cerebrovascular disease, which together shape cognitive trajectories. This interacts directly with genetics: carrying the APOE e4 allele raises AD risk more in women than in men, meaning the same genetic risk factor confers different real-world resilience outcomes depending on sex -- and resilience itself appears to shift across different ages and disease stages rather than being fixed.",
        "supporting_sources": ["pubmed_38967222", "wikipedia_apolipoprotein_e", "pubmed_33340485"],
        "reasoning_path": ["SexDifferences", "APOE_e4", "AD", "AmyloidBeta", "TauProtein"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q27",
        "question": "How do the glymphatic system, sleep, and aging interact to form a proposed 'final common pathway' to dementia, and how does this relate to the amyloid and tau aggregation seen in Alzheimer's disease specifically?",
        "type": "synthesis",
        "expected_answer": "The glymphatic system clears the brain of protein waste products and is mostly active during sleep, but it degrades with age -- so aging and poor sleep compound to reduce clearance capacity. Since this system's role is clearing aggregation-prone proteins, its failure is proposed as a 'final common pathway' across neurodegenerative dementias, including Alzheimer's, where impaired clearance would be expected to worsen the buildup of amyloid-beta plaques and hyperphosphorylated tau tangles that define AD's core pathology.",
        "supporting_sources": ["pubmed_33004510", "wikipedia_amyloid_beta", "wikipedia_tau_protein"],
        "reasoning_path": ["GlymphaticSystem", "SleepDisturbance", "Aging", "AmyloidBeta", "TauProtein", "Dementia"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q28",
        "question": "What is the overall societal impact of Alzheimer's disease when considering economic burden, caregiver burden, and epidemiological prevalence trends together?",
        "type": "synthesis",
        "expected_answer": "AD prevalence continues to rise with global population aging, and the disease is expected to remain the leading cause of dementia (50-70% of cases). This growing prevalence directly drives cost: the global cost of dementia care is projected to reach roughly $2 trillion by 2030, with indirect caregiving costs (largely borne by family caregivers) as the single largest cost contributor for community-dwelling patients, while formal caregiving accommodation shifts costs toward direct expenditure. Together, rising prevalence and disproportionate caregiving burden make AD an escalating public health and economic priority, reinforcing the case for early prevention efforts targeting modifiable risk factors.",
        "supporting_sources": ["pubmed_37972428", "pubmed_34101789", "wikipedia_dementia"],
        "reasoning_path": ["EconomicBurden", "CaregiverBurden", "AD", "Aging"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q29",
        "question": "How does the StatPearls textbook's staged clinical and pathological description of Alzheimer's disease align with the biomarker-based findings on amyloid, tau, and imaging described in the PubMed literature?",
        "type": "synthesis",
        "expected_answer": "The textbook frames AD through etiology, pathophysiology, histopathology, staging, and prognosis sections that describe progressive amyloid plaque and neurofibrillary tangle accumulation alongside clinical decline. This matches the PubMed literature's biomarker-based framing: fluid biomarkers (CSF/plasma amyloid-beta and tau) and imaging biomarkers (amyloid PET, tau PET, structural MRI) are used to detect the same underlying amyloid/tau pathology described in the textbook, but push detection earlier -- into a 'preclinical' stage before clinical symptoms appear -- refining the textbook's clinical staging with molecular and imaging precision.",
        "supporting_sources": ["textbook_statpearls_alzheimer_disease", "pubmed_34456336", "pubmed_26921134", "pubmed_26827786"],
        "reasoning_path": ["AmyloidPlaques", "NFT", "FluidBiomarkers", "AmyloidPET", "TauPET", "MRIAtrophy", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
    {
        "question_id": "q30",
        "question": "Based on this corpus, what would a comprehensive Alzheimer's prevention strategy look like, considering both genetic risk and modifiable risk factors?",
        "type": "synthesis",
        "expected_answer": "A comprehensive strategy would combine early genetic risk stratification (APOE genotyping to identify e4 carriers at elevated risk, with awareness that risk is sex-modified, and family-history screening given the 10-30% risk increase with an affected first-degree relative) with modifiable-factor interventions started in midlife or earlier: promoting physical activity (a consistently protective factor), managing cardiovascular/vascular risk factors, reducing environmental exposures like air pollution, and protecting sleep quality to preserve glymphatic clearance. Because AD is described as starting decades before clinical symptoms, this combined genetic-plus-lifestyle approach targeting non-demented middle-aged and elderly populations offers the greatest potential to prevent or delay onset, complementing emerging disease-modifying drug therapies rather than replacing them.",
        "supporting_sources": ["pubmed_33340485", "textbook_statpearls_alzheimer_disease", "pubmed_34101789", "pubmed_35301183", "pubmed_39889875", "pubmed_33004510"],
        "reasoning_path": ["APOE_e4", "FamilyHistory", "PhysicalActivity", "CardiovascularRisk", "AirPollution", "SleepDisturbance", "AD"],
        "difficulty": "hard",
        "designed_to_favor": "hybridrag",
    },
]


def main():
    by_favor = {}
    for qa in QA_PAIRS:
        by_favor.setdefault(qa["designed_to_favor"], 0)
        by_favor[qa["designed_to_favor"]] += 1

    dataset = {
        "dataset_name": "alzheimers_hybridrag_ablation_eval",
        "description": "30-item QA evaluation set for ablation study comparing VectorRAG, GraphRAG, and HybridRAG over a 31-document Alzheimer's corpus (20 PubMed abstracts + 10 Wikipedia articles + 1 StatPearls textbook chapter).",
        "num_questions": len(QA_PAIRS),
        "distribution_by_target_method": by_favor,
        "usage_notes": [
            "designed_to_favor is a hypothesis about which retrieval method should perform best on that question, not a guarantee -- use it to check whether ablation results match expectations.",
            "reasoning_path lists Knowledge graph/nodes.csv node IDs relevant to the question; useful for scoring GraphRAG's retrieved subgraph against ground truth.",
            "supporting_sources lists article_id values matching files in 'Selected articles/' and 'chunking/' -- useful for scoring VectorRAG's retrieved chunks against ground truth.",
            "expected_answer is a synthesized gold reference, not a verbatim quote from any single source.",
        ],
        "questions": QA_PAIRS,
    }

    out_path = OUT_DIR / "eval_dataset.json"
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(QA_PAIRS)} QA pairs -> {out_path}")
    print(f"Distribution: {by_favor}")


if __name__ == "__main__":
    main()
