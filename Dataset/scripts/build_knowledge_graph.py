"""
Build a well-connected Alzheimer's knowledge graph grounded in the 31 fetched
sources (20 PubMed abstracts + 10 Wikipedia articles + 1 StatPearls textbook
chapter). Entities and relations were curated by reading the actual source
text (see 'Selected articles/') so every edge cites the article(s) it is
grounded in. Evidence strings are paraphrased summaries, not verbatim quotes.

Outputs Neo4j-admin-import-ready CSVs:
  Knowledge graph/nodes.csv   (nodeId:ID, name, type:LABEL, description)
  Knowledge graph/edges.csv   (:START_ID, :END_ID, :TYPE, source_articles, evidence)
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent.parent / "Knowledge graph"
OUT_DIR.mkdir(exist_ok=True)

# id, name, type, description
NODES = [
    ("AD", "Alzheimer's Disease", "Disease", "Most common cause of dementia worldwide; biologically defined by amyloid plaques and neurofibrillary tangles"),
    ("Dementia", "Dementia", "Disease", "Umbrella syndrome of acquired cognitive decline affecting daily function"),
    ("VascularDementia", "Vascular Dementia", "Disease", "Second most common dementia subtype, driven by cerebrovascular disease"),
    ("MixedDementia", "Mixed Dementia", "Disease", "Co-occurrence of more than one dementia pathology, most often AD plus vascular dementia"),
    ("MCI", "Mild Cognitive Impairment", "Disease", "Prodromal cognitive decline stage; amnestic MCI often precedes AD"),
    ("DownSyndrome", "Down Syndrome", "Disease", "Trisomy 21 condition; extra APP gene copy links it to the amyloid hypothesis"),
    ("AmyloidBeta", "Amyloid-beta (Abeta)", "Protein", "Peptide fragment of APP that aggregates into amyloid plaques; core AD biomarker"),
    ("APP", "Amyloid Precursor Protein / APP gene", "Protein_Gene", "Transmembrane protein cleaved to release amyloid-beta; gene on chromosome 21"),
    ("TauProtein", "Tau Protein", "Protein", "Microtubule-stabilizing protein; hyperphosphorylated form aggregates into tangles"),
    ("NFT", "Neurofibrillary Tangles", "Pathology", "Intracellular aggregates of hyperphosphorylated tau; hallmark AD pathology"),
    ("AmyloidPlaques", "Amyloid Plaques", "Pathology", "Extracellular aggregates of amyloid-beta fibrils; hallmark AD pathology"),
    ("APOE", "Apolipoprotein E (APOE)", "Protein_Gene", "Lipid-transport protein/gene; allele status is the major genetic risk modifier for sporadic AD"),
    ("APOE_e4", "APOE epsilon-4 allele", "GeneVariant", "Strongest common genetic risk factor for sporadic AD"),
    ("APOE_e2", "APOE epsilon-2 allele", "GeneVariant", "Strongest common genetic protective factor against AD"),
    ("PSEN1", "Presenilin 1 (PSEN1)", "Gene", "Gene on chromosome 14; mutations cause autosomal-dominant familial AD"),
    ("PSEN2", "Presenilin 2 (PSEN2)", "Gene", "Gene on chromosome 1; mutations cause autosomal-dominant familial AD"),
    ("Chr21", "Chromosome 21", "Chromosome", "Carries the APP gene; triplicated in Down syndrome"),
    ("Chr14", "Chromosome 14", "Chromosome", "Carries the PSEN1 gene"),
    ("Chr1", "Chromosome 1", "Chromosome", "Carries the PSEN2 gene"),
    ("Hippocampus", "Hippocampus", "BrainRegion", "Brain region for memory formation; earliest major atrophy site in AD"),
    ("Microglia", "Microglia", "CellType", "Brain innate-immune cells; mediate neuroinflammation and interact with Abeta and tau"),
    ("Neuroinflammation", "Neuroinflammation", "Mechanism", "Innate immune activation implicated in AD pathogenesis, partly independent of amyloid"),
    ("GlymphaticSystem", "Glymphatic System", "Mechanism", "Brain waste-clearance system, most active during sleep; degrades with age"),
    ("SleepDisturbance", "Sleep Disturbance", "RiskFactor", "Impaired sleep architecture is a frequent antecedent of dementia onset"),
    ("SynapticDysfunction", "Synaptic Dysfunction / Loss", "Mechanism", "Loss of synaptic homeostasis central to AD-associated cognitive decline"),
    ("AmyloidCascadeHypothesis", "Amyloid Cascade Hypothesis", "Theory", "Hypothesis that amyloid-beta accumulation triggers downstream tau pathology and neurodegeneration"),
    ("NMDAReceptor", "NMDA Receptor", "MolecularTarget", "Glutamate receptor; pathological overstimulation implicated in excitotoxic neurodegeneration"),
    ("GlutamateExcitotoxicity", "Glutamate Excitotoxicity", "Mechanism", "Pathological NMDA receptor overstimulation by glutamate, hypothesized to contribute to AD"),
    ("Acetylcholine", "Acetylcholine", "Neurotransmitter", "Neurotransmitter reduced in the AD brain; target of cholinesterase inhibitor therapy"),
    ("CholinesteraseInhibitors", "Cholinesterase Inhibitors", "DrugClass", "Main symptomatic AD drug class; raises synaptic acetylcholine"),
    ("Donepezil", "Donepezil", "Drug", "Reversible cholinesterase inhibitor used for symptomatic AD treatment"),
    ("Rivastigmine", "Rivastigmine", "Drug", "Pseudo-irreversible cholinesterase inhibitor used for AD"),
    ("Galantamine", "Galantamine", "Drug", "Reversible cholinesterase inhibitor, less well tolerated than donepezil"),
    ("Tacrine", "Tacrine", "Drug", "First licensed cholinesterase inhibitor; withdrawn due to hepatotoxicity"),
    ("Hepatotoxicity", "Hepatotoxicity", "AdverseEffect", "Liver toxicity that led to tacrine's discontinuation"),
    ("Memantine", "Memantine", "Drug", "NMDA receptor antagonist used for moderate-to-severe AD"),
    ("AntiAmyloidAntibodies", "Anti-Amyloid Monoclonal Antibodies", "DrugClass", "Disease-modifying antibody class that clears brain amyloid-beta"),
    ("Lecanemab", "Lecanemab", "Drug", "FDA-approved (2023) anti-amyloid antibody; slows cognitive decline, can cause ARIA"),
    ("Aducanumab", "Aducanumab", "Drug", "FDA accelerated-approval anti-amyloid antibody for early AD"),
    ("ARIA", "Amyloid-Related Imaging Abnormalities (ARIA)", "AdverseEffect", "Brain edema/microhemorrhage side effect of anti-amyloid antibody therapy"),
    ("FluidBiomarkers", "CSF / Plasma Biomarkers (Abeta42, tau)", "Biomarker", "Cerebrospinal fluid or plasma amyloid-beta and tau levels used for AD diagnosis"),
    ("AmyloidPET", "Amyloid PET Imaging", "Biomarker", "In-vivo imaging technique that detects brain amyloid plaque burden"),
    ("TauPET", "Tau PET Imaging", "Biomarker", "Imaging technique for tau pathology; aids differential diagnosis and trial selection"),
    ("MRIAtrophy", "Structural MRI (atrophy)", "Biomarker", "Detects hippocampal and cortical atrophy for diagnosis and disease tracking"),
    ("PhysicalActivity", "Physical Activity", "ProtectiveFactor", "Associated with a meaningfully lower risk of AD and all-cause dementia"),
    ("CardiovascularRisk", "Cardiovascular / Vascular Risk Factors", "RiskFactor", "Hypertension and vascular disease raise dementia and AD risk"),
    ("AirPollution", "Air Pollution (particulate matter)", "RiskFactor", "Fine particulate matter and related pollutants raise dementia and AD risk"),
    ("Aging", "Aging", "RiskFactor", "The single greatest risk factor for AD and dementia generally"),
    ("FamilyHistory", "Family History / Genetic Predisposition", "RiskFactor", "A first-degree relative with AD raises personal risk substantially"),
    ("CaregiverBurden", "Caregiver Burden", "SocialImpact", "Indirect caregiving cost is the main driver of AD's societal cost"),
    ("EconomicBurden", "Economic Burden of AD", "SocialImpact", "Global dementia care cost is projected to reach roughly $2 trillion by 2030"),
    ("SexDifferences", "Sex / Gender Differences", "ModifyingFactor", "Biological sex and gender modify AD risk, pathology burden, and resilience"),
    ("MemoryLoss", "Memory Loss", "Symptom", "Short-term memory loss is typically the earliest noticeable AD symptom"),
    ("CognitiveDecline", "Cognitive Decline", "Symptom", "Progressive decline in cognition and function across AD stages"),
]

# start_id, end_id, relation, [article_ids], evidence (paraphrased)
EDGES = [
    ("AmyloidPlaques", "AD", "HALLMARK_OF", ["pubmed_30135715", "pubmed_33986301"], "AD is biologically defined by beta-amyloid plaques together with tau neurofibrillary tangles."),
    ("NFT", "AD", "HALLMARK_OF", ["pubmed_30135715", "pubmed_33986301", "pubmed_21371747"], "Neurofibrillary tangles of hyperphosphorylated tau are one of AD's two classic hallmark pathologies."),
    ("AmyloidBeta", "AmyloidPlaques", "AGGREGATES_INTO", ["wikipedia_amyloid_beta"], "Amyloid-beta oligomers aggregate into fibrils that form the plaques seen in AD brains."),
    ("TauProtein", "NFT", "HYPERPHOSPHORYLATED_TO_FORM", ["wikipedia_tau_protein"], "Hyperphosphorylation converts normally microtubule-stabilizing tau into insoluble tangle aggregates."),
    ("NFT", "TauProtein", "COMPOSED_OF", ["wikipedia_neurofibrillary_tangle"], "Neurofibrillary tangles are composed largely of paired helical filaments of hyperphosphorylated tau, distinct from the straight-filament tau seen in other tauopathies."),
    ("APP", "AmyloidBeta", "CLEAVED_TO_PRODUCE", ["pubmed_31753135"], "Proteolytic cleavage of APP produces the amyloid-beta fragment central to AD pathology."),
    ("APP", "Chr21", "LOCATED_ON", ["textbook_statpearls_alzheimer_disease"], "The APP gene resides on chromosome 21."),
    ("PSEN1", "Chr14", "LOCATED_ON", ["textbook_statpearls_alzheimer_disease"], "PSEN1 is located on chromosome 14."),
    ("PSEN2", "Chr1", "LOCATED_ON", ["textbook_statpearls_alzheimer_disease"], "PSEN2 is located on chromosome 1."),
    ("DownSyndrome", "Chr21", "EXTRA_COPY_OF", ["textbook_statpearls_alzheimer_disease"], "Trisomy 21 gives an extra copy of chromosome 21 and therefore an extra APP gene copy."),
    ("DownSyndrome", "AmyloidCascadeHypothesis", "SUPPORTS_EVIDENCE_FOR", ["textbook_statpearls_alzheimer_disease"], "The APP triplication in Down syndrome is a key line of evidence for the amyloid hypothesis."),
    ("DownSyndrome", "AD", "ELEVATES_RISK_OF", ["textbook_statpearls_alzheimer_disease"], "Extra APP dosage in Down syndrome markedly raises lifetime AD risk."),
    ("PSEN1", "AD", "MUTATION_CAUSES", ["pubmed_23276979"], "PSEN1 mutations are one of three known causes of autosomal-dominant familial AD."),
    ("PSEN2", "AD", "MUTATION_CAUSES", ["pubmed_23276979"], "PSEN2 mutations are one of three known causes of autosomal-dominant familial AD."),
    ("APP", "AD", "MUTATION_CAUSES", ["pubmed_23276979"], "APP mutations are one of three known causes of autosomal-dominant familial AD."),
    ("APOE_e4", "APOE", "ALLELE_OF", ["wikipedia_apolipoprotein_e"], "APOE has three major alleles; epsilon-4 is one of them."),
    ("APOE_e2", "APOE", "ALLELE_OF", ["wikipedia_apolipoprotein_e"], "APOE has three major alleles; epsilon-2 is one of them."),
    ("APOE_e4", "AD", "STRONGEST_GENETIC_RISK_FACTOR_FOR", ["pubmed_33340485"], "APOE epsilon-4 remains the strongest genetic risk factor for sporadic AD after large GWAS meta-analyses."),
    ("APOE_e2", "AD", "PROTECTIVE_FACTOR_FOR", ["pubmed_33340485"], "APOE epsilon-2 is the strongest known genetic protective factor against AD."),
    ("APOE", "AmyloidBeta", "INTERACTS_WITH", ["pubmed_33340485"], "APOE was first linked to AD via its effect on amyloid-beta aggregation and clearance."),
    ("APOE", "TauProtein", "INTERACTS_WITH", ["pubmed_33340485"], "APOE research has expanded to include effects on tau neurofibrillary degeneration."),
    ("APOE", "Microglia", "MODULATES", ["pubmed_33340485"], "APOE genotype shapes microglial and astrocyte inflammatory responses in AD."),
    ("APOE_e4", "SexDifferences", "RISK_MODIFIED_BY", ["wikipedia_apolipoprotein_e"], "Carrying APOE4 raises AD risk more in women than in men."),
    ("SexDifferences", "AD", "MODIFIES_RISK_OF", ["pubmed_38967222"], "Sex and gender jointly shape AD risk-factor prevalence, pathology burden, and cognitive resilience."),
    ("FamilyHistory", "AD", "RISK_FACTOR_FOR", ["textbook_statpearls_alzheimer_disease"], "A first-degree relative with AD raises personal risk by roughly 10-30%."),
    ("Aging", "AD", "RISK_FACTOR_FOR", ["pubmed_21371747", "wikipedia_dementia"], "Aging is the greatest single risk factor for AD and dementia generally."),
    ("Aging", "GlymphaticSystem", "DEGRADES", ["pubmed_33004510"], "Glymphatic clearance efficiency declines with age."),
    ("SleepDisturbance", "Dementia", "PRECEDES_ONSET_OF", ["pubmed_33004510"], "Disrupted sleep architecture frequently precedes clinical dementia onset."),
    ("SleepDisturbance", "GlymphaticSystem", "IMPAIRS", ["pubmed_33004510"], "Poor sleep reduces the glymphatic system's brain-clearance activity, which is greatest during sleep."),
    ("GlymphaticSystem", "AmyloidBeta", "CLEARS", ["pubmed_33004510"], "The glymphatic system is a proposed clearance route for amyloid-beta and other aggregation-prone proteins."),
    ("PhysicalActivity", "AD", "PROTECTIVE_FACTOR_FOR", ["pubmed_35301183"], "Meta-analysis found physical activity associated with a meaningfully lower AD incidence."),
    ("PhysicalActivity", "Dementia", "PROTECTIVE_FACTOR_FOR", ["pubmed_35301183"], "Physical activity is also associated with lower all-cause and vascular dementia incidence."),
    ("CardiovascularRisk", "VascularDementia", "RISK_FACTOR_FOR", ["pubmed_39889875", "wikipedia_dementia"], "Vascular and cardiovascular risk factors are the primary drivers of vascular dementia."),
    ("CardiovascularRisk", "AD", "RISK_FACTOR_FOR", ["pubmed_34101789"], "Vascular risk factors are among the modifiable contributors to AD risk."),
    ("AirPollution", "AD", "RISK_FACTOR_FOR", ["pubmed_39889875"], "Fine particulate matter and related pollutants were associated with higher AD dementia risk in an umbrella review."),
    ("AirPollution", "VascularDementia", "RISK_FACTOR_FOR", ["pubmed_39889875"], "Fine particulate matter and chronic noise were associated with higher vascular dementia risk."),
    ("VascularDementia", "AD", "COMORBID_WITH", ["wikipedia_dementia"], "AD is frequently diagnosed alongside vascular dementia as mixed dementia."),
    ("MixedDementia", "AD", "COMBINES", ["wikipedia_dementia"], "The most common mixed-dementia presentation combines AD with vascular dementia."),
    ("MixedDementia", "VascularDementia", "COMBINES", ["wikipedia_dementia"], "The most common mixed-dementia presentation combines AD with vascular dementia."),
    ("AD", "Dementia", "SUBTYPE_OF", ["wikipedia_dementia"], "AD accounts for roughly 60-70% of all dementia cases worldwide."),
    ("VascularDementia", "Dementia", "SUBTYPE_OF", ["wikipedia_dementia"], "Vascular dementia is the second most common dementia subtype."),
    ("MCI", "AD", "PRODROMAL_STAGE_OF", ["wikipedia_mild_cognitive_impairment"], "Amnestic MCI is frequently a prodromal stage that converts to AD at roughly 10-15% per year."),
    ("Hippocampus", "AD", "ATROPHIES_IN", ["pubmed_26827786"], "Hippocampal volumetry and atrophy tracking are established outcome markers in AD research."),
    ("Hippocampus", "MemoryLoss", "SUPPORTS_FUNCTION_OF", ["wikipedia_alzheimers_disease"], "AD targets the hippocampus, the brain region responsible for memory formation."),
    ("MRIAtrophy", "Hippocampus", "MEASURES", ["pubmed_26827786"], "Structural MRI is the primary tool for quantifying hippocampal and cortical atrophy in AD."),
    ("MRIAtrophy", "AD", "BIOMARKER_FOR", ["pubmed_26921134"], "MRI is an established imaging technique for AD diagnosis and progression tracking."),
    ("AmyloidPET", "AmyloidPlaques", "DETECTS", ["pubmed_26921134", "wikipedia_amyloid_beta"], "Amyloid PET imaging visualizes brain amyloid plaque burden in vivo."),
    ("TauPET", "NFT", "DETECTS", ["pubmed_26921134"], "Tau PET imaging visualizes neurofibrillary tangle burden and aids differential diagnosis."),
    ("FluidBiomarkers", "AmyloidBeta", "MEASURES", ["pubmed_26921134"], "CSF amyloid-beta42 is an established core fluid biomarker for AD."),
    ("FluidBiomarkers", "TauProtein", "MEASURES", ["pubmed_26921134"], "CSF tau is an established core fluid biomarker for AD."),
    ("FluidBiomarkers", "AD", "BIOMARKER_FOR", ["pubmed_34456336"], "Amyloid-beta pathway biomarkers underpin modern biological definitions of AD."),
    ("FluidBiomarkers", "MCI", "DIAGNOSTIC_FOR", ["pubmed_38849944"], "A low plasma Abeta42/40 ratio strongly supported AD diagnosis in an MCI/early-dementia trial cohort."),
    ("MRIAtrophy", "MCI", "DETECTS_PROGRESSION_IN", ["wikipedia_mild_cognitive_impairment"], "MRI can track progressive gray-matter loss as MCI converts toward full AD dementia."),
    ("AmyloidCascadeHypothesis", "AD", "EXPLAINS", ["pubmed_36911732"], "The amyloid cascade hypothesis has long been the leading model of AD pathogenesis."),
    ("Neuroinflammation", "AmyloidCascadeHypothesis", "CHALLENGES", ["pubmed_36911732"], "Evidence for neuroinflammation's independent role has weakened the pure amyloid-centric model."),
    ("Microglia", "Neuroinflammation", "MEDIATES", ["pubmed_36911732"], "Microglia-mediated responses are the primary drivers of AD-associated neuroinflammation."),
    ("Microglia", "AmyloidBeta", "INTERACTS_WITH", ["pubmed_36911732"], "Microglia physically interact with amyloid-beta deposits as part of the innate immune response."),
    ("Microglia", "TauProtein", "INTERACTS_WITH", ["pubmed_36911732"], "Microglia also interact with tau pathology, influencing its spread and clearance."),
    ("Neuroinflammation", "AD", "CONTRIBUTES_TO", ["pubmed_36911732"], "Elevated inflammatory markers and innate-immune AD risk genes point to neuroinflammation as a core disease driver."),
    ("SynapticDysfunction", "AD", "CENTRAL_TO", ["pubmed_33986301"], "Loss of synaptic homeostasis is framed as central to AD's biological definition."),
    ("AmyloidBeta", "SynapticDysfunction", "CAUSES", ["pubmed_31753135"], "Amyloid-beta accumulation contributes to synaptic strength reduction and synapse loss."),
    ("TauProtein", "SynapticDysfunction", "CAUSES", ["pubmed_31753135"], "Hyperphosphorylated tau aggregation contributes to synaptic strength reduction and synapse loss."),
    ("SynapticDysfunction", "CognitiveDecline", "LEADS_TO", ["pubmed_33986301"], "Progressive synaptic loss underlies the cognitive decline seen across AD stages."),
    ("MemoryLoss", "AD", "EARLY_SYMPTOM_OF", ["wikipedia_alzheimers_disease"], "Short-term memory loss is typically the most noticeable early AD symptom."),
    ("CognitiveDecline", "AD", "SYMPTOM_OF", ["pubmed_31753135"], "Progressive cognitive and functional decline defines the clinical course of AD."),
    ("GlutamateExcitotoxicity", "AD", "CONTRIBUTES_TO", ["wikipedia_memantine"], "Glutamatergic excitotoxicity is hypothesized to contribute to AD's etiology."),
    ("NMDAReceptor", "GlutamateExcitotoxicity", "OVERSTIMULATED_IN", ["pubmed_12672860"], "Pathological NMDA receptor overstimulation by glutamate drives excitotoxic neurodegeneration."),
    ("Memantine", "NMDAReceptor", "ANTAGONIST_OF", ["pubmed_12672860", "wikipedia_memantine"], "Memantine is an uncompetitive NMDA receptor antagonist."),
    ("Memantine", "AD", "TREATS", ["pubmed_12672860"], "A placebo-controlled trial found memantine reduced clinical deterioration in moderate-to-severe AD."),
    ("CholinesteraseInhibitors", "Acetylcholine", "INCREASES", ["wikipedia_cholinesterase_inhibitor"], "Cholinesterase inhibitors block acetylcholine breakdown, raising synaptic acetylcholine levels."),
    ("CholinesteraseInhibitors", "AD", "TREATS", ["pubmed_33035532"], "Cholinesterase inhibitors are the main class of drugs used for symptomatic AD treatment."),
    ("Donepezil", "CholinesteraseInhibitors", "MEMBER_OF_CLASS", ["pubmed_33035532", "wikipedia_cholinesterase_inhibitor"], "Donepezil is one of the three cholinesterase inhibitors currently on the market for AD."),
    ("Rivastigmine", "CholinesteraseInhibitors", "MEMBER_OF_CLASS", ["pubmed_33035532", "wikipedia_cholinesterase_inhibitor"], "Rivastigmine is one of the three cholinesterase inhibitors currently on the market for AD."),
    ("Galantamine", "CholinesteraseInhibitors", "MEMBER_OF_CLASS", ["pubmed_33035532", "wikipedia_cholinesterase_inhibitor"], "Galantamine is one of the three cholinesterase inhibitors currently on the market for AD."),
    ("Tacrine", "CholinesteraseInhibitors", "MEMBER_OF_CLASS", ["pubmed_33035532"], "Tacrine was the first licensed cholinesterase inhibitor for AD."),
    ("Tacrine", "Hepatotoxicity", "DISCONTINUED_DUE_TO", ["pubmed_33035532"], "Tacrine was withdrawn from clinical use because of hepatotoxicity."),
    ("Donepezil", "MCI", "MINOR_BENEFIT_FOR", ["wikipedia_mild_cognitive_impairment"], "Donepezil showed only minor, short-term cognitive benefit in MCI trials, with notable side effects."),
    ("Rivastigmine", "MCI", "NO_BENEFIT_FOR", ["wikipedia_mild_cognitive_impairment"], "Rivastigmine failed to slow progression from MCI to AD in trials."),
    ("AntiAmyloidAntibodies", "AmyloidPlaques", "REDUCES", ["pubmed_37955845"], "Anti-amyloid antibodies produce a marked reduction in total brain amyloid burden on PET."),
    ("AntiAmyloidAntibodies", "CognitiveDecline", "SLOWS", ["pubmed_37955845"], "Anti-amyloid antibody trials achieved roughly a 30% slowing of cognitive decline."),
    ("Lecanemab", "AntiAmyloidAntibodies", "MEMBER_OF_CLASS", ["pubmed_37955845"], "Lecanemab is one of the anti-amyloid monoclonal antibodies approved for early AD."),
    ("Aducanumab", "AntiAmyloidAntibodies", "MEMBER_OF_CLASS", ["pubmed_37955845"], "Aducanumab is one of the anti-amyloid monoclonal antibodies approved for early AD."),
    ("Lecanemab", "AD", "TREATS", ["wikipedia_lecanemab"], "Lecanemab received full FDA approval in 2023 for treating early Alzheimer's disease."),
    ("Aducanumab", "AD", "TREATS", ["pubmed_37955845"], "Aducanumab received accelerated FDA approval for treating early Alzheimer's disease."),
    ("Lecanemab", "ARIA", "CAUSES", ["wikipedia_lecanemab"], "Lecanemab treatment can cause amyloid-related imaging abnormalities (brain edema or microhemorrhage)."),
    ("ARIA", "AntiAmyloidAntibodies", "SIDE_EFFECT_OF", ["pubmed_37955845"], "ARIA monitoring via MRI is a standard safety requirement across anti-amyloid antibody therapies."),
    ("CaregiverBurden", "AD", "RESULTS_FROM", ["pubmed_37972428"], "Indirect caregiving costs are the largest single contributor to AD's societal cost."),
    ("EconomicBurden", "AD", "RESULTS_FROM", ["pubmed_37972428"], "The global cost of AD-related dementia care is projected to reach roughly $2 trillion by 2030."),
    ("CaregiverBurden", "EconomicBurden", "CONTRIBUTES_TO", ["pubmed_37972428"], "Indirect caregiving cost is the main driver of AD's overall societal economic burden."),
]


def main():
    node_ids = {n[0] for n in NODES}
    for s, e, *_ in EDGES:
        if s not in node_ids:
            raise ValueError(f"Edge references unknown node: {s}")
        if e not in node_ids:
            raise ValueError(f"Edge references unknown node: {e}")

    nodes_path = OUT_DIR / "nodes.csv"
    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nodeId:ID", "name", "type:LABEL", "description"])
        for node_id, name, ntype, desc in NODES:
            writer.writerow([node_id, name, ntype, desc])

    edges_path = OUT_DIR / "edges.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID", ":END_ID", ":TYPE", "source_articles", "evidence"])
        for start, end, rel, sources, evidence in EDGES:
            writer.writerow([start, end, rel, ";".join(sources), evidence])

    # connectivity check: every node should touch at least one edge
    touched = set()
    for s, e, *_ in EDGES:
        touched.add(s)
        touched.add(e)
    isolated = node_ids - touched

    print(f"Nodes: {len(NODES)}  Edges: {len(EDGES)}")
    print(f"Isolated (no edges) nodes: {sorted(isolated) if isolated else 'none'}")

    # coverage check: how many of the 31 source articles are cited by >=1 edge
    all_cited = set()
    for _, _, _, sources, _ in EDGES:
        all_cited.update(sources)
    print(f"Distinct source articles cited across edges: {len(all_cited)}")


if __name__ == "__main__":
    main()
