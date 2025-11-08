# Precision Oncology Chemotherapy Sensitivity Predictor

**HTHSCI 2E03 Inquiry Biochemistry - McMaster University**

## Our Project: Understanding Chemotherapy Resistance

Our group is investigating **mechanisms of chemotherapy resistance** in non-small cell lung cancer (NSCLC). This interactive tool serves as our research platform, allowing us to test hypotheses about resistance patterns using real clinical data from 2,934 treatment records across 1,060 patients.

### Why This Tool Supports Our Research

**1. Test Resistance Hypotheses Directly**
We can input specific mutation profiles and observe predicted treatment outcomes, validating whether genomic alterations like TP53 loss or KRAS mutations correlate with poor chemotherapy response in real patient data.

**2. Compare Intrinsic vs. Acquired Resistance**
The stratified models separate first-line (intrinsic resistance) from later-line treatments (acquired resistance), letting us quantify how resistance burden accumulates with each treatment cycle.

**3. Analyze Drug Class-Specific Mechanisms**
We examine whether platinum agents, taxanes, and targeted inhibitors show distinct resistance profiles based on their mechanisms—DNA crosslinking vs. microtubule disruption vs. kinase inhibition.

**4. Generate Evidence for Our Presentation**
Feature importance analysis reveals which mutations most strongly predict resistance, providing data-driven conclusions about p53-mediated apoptosis defects, EGFR pathway dependencies, and drug efflux mechanisms.

**5. Demonstrate Clinical Relevance**
We show how biochemical resistance mechanisms translate to measurable outcomes (PFS, resistance probability), connecting molecular biology to patient survival data.

### Our Research Questions

1. **Do TP53 mutations confer universal chemotherapy resistance?** We predict outcomes across multiple drug classes with and without TP53 mutations to determine if apoptosis defects broadly reduce treatment efficacy.

2. **Why do EGFR inhibitors fail after initial response?** We model sequential treatment lines to identify whether resistance emerges from secondary mutations (T790M) or pathway redundancy (MET amplification).

3. **Can combination therapy overcome single-agent resistance?** We compare monotherapy vs. combination regimens to quantify whether multi-target approaches reduce the 6-month resistance probability.

4. **How does treatment history affect subsequent therapy?** We analyze the same patient profile at different treatment lines to measure acquired resistance accumulation.

### Novel Screening & Treatment Strategies From Our Research

Our investigation extends beyond traditional genomic biomarkers to propose cutting-edge approaches for predicting and overcoming chemotherapy resistance:

**Advanced Biomarker Discovery:**
- **Notch Pathway Mutations:** We examine how deleterious NOTCH1/2/3 mutations (del-NOTCH mut) identified via computational tools like PolyPhen-2 predict immunotherapy response. Using the tool's predictions for immunotherapy agents (pembrolizumab, nivolumab), we validate whether Notch-mutant profiles show improved PFS with checkpoint inhibitors versus traditional chemotherapy.
  
- **HES1 Expression:** Since HES1 (a Notch target gene) mediates EGFR-TKI resistance, we model gefitinib/erlotinib/osimertinib responses across treatment lines to identify where HES1-driven resistance mechanisms likely emerge, guiding when to screen for this biomarker clinically.

**Metabolic Resistance Mechanisms:**
- **KRAS/STK11/MCT4 Axis:** Our tool allows testing KRAS-mutant adenocarcinoma responses to platinum-based chemotherapy. Poor predicted PFS in KRAS+ cases supports our hypothesis that STK11 inactivation enhances lactate production via MCT4, creating metabolic resistance. This validates MCT4 transporter knockout as a potential therapeutic target.

- **Lipid Metabolism in ALK+ Tumors:** We model ALK inhibitor resistance across treatment lines, providing evidence that metabolic reprogramming (SREBP-1 activation) contributes to acquired resistance observed in later-line predictions.

**Microbiome-Guided Treatment Selection:**
- **Gut Microbiota Profiling:** Using predictions for platinum + pemetrexed or gemcitabine combinations, we correlate literature-identified bacterial signatures (*Streptococcus mutans*, *Bacteroides intestinalis*) with predicted response patterns. This bridges our machine learning model with emerging microbiome sequencing as a novel screening tool.

**Therapeutic Intervention Strategies:**
- **ADAM17 Inhibition:** Since ADAM17 cleavage activates Notch signaling, we propose inhibiting ADAM17 to sensitize resistant tumors. Our tool models combination therapy scenarios (e.g., gefitinib + hypothetical ADAM17 inhibitor) to demonstrate potential PFS improvements.

- **Sequential TKI Therapy:** We generate case studies showing first-generation EGFR-TKI response at line 1, acquired resistance (T790M) at line 2, and third-generation TKI (osimertinib) rescue at line 3, validating genotype-guided sequential therapy.

**Machine Learning Integration:**
- **Multi-Omics Data Fusion:** Our stratified models (XGBoost + Deep Learning) demonstrate how AI integrates genomic, clinical, and treatment history data to predict resistance. This parallels literature approaches using 117 ML models to identify prognostic genes (TBCD, PTPRC, LDHA, ACTR2).

- **LDHA as Predictive Biomarker:** High LDHA expression predicts immunotherapy resistance in literature. We test whether the tool's genomic features capture LDHA-related metabolic signatures that correlate with poor predicted response to checkpoint inhibitors.

### Validation of Our Hypotheses

**Supported by Tool Predictions:**
1. **EGFR+ tumors respond better to gefitinib than platinum-based chemotherapy** (targetable vulnerability)
2. **TP53 loss correlates with decreased predicted PFS** (apoptosis defect)
3. **Later treatment lines show progressively worse outcomes** (acquired resistance burden)
4. **Combination regimens reduce resistance probability** (multi-target approach)

**Requiring External Validation:**
- Notch pathway mutations (if not in 1,318 GENIE features, we propose adding via literature data)
- Gut microbiome profiles (propose integrating metagenomic data with clinical predictions)
- Telomere length measurements (leukocyte RTL predicts osimertinib adverse reactions)

### Our Methodology

We use this tool to:
- **Generate predictions** for standardized patient profiles across 81 chemotherapy agents
- **Extract SHAP values** to identify genomic features driving resistance
- **Compare confidence scores** to assess prediction certainty near the clinical benefit threshold (6 months PFS)
- **Document case studies** demonstrating resistance mechanisms for our presentation

This data-driven approach grounds our biochemistry concepts in actual clinical outcomes, showing how molecular alterations manifest as treatment failure.

---

## Educational Purpose

This interactive web application was developed as an educational tool for **HTHSCI 2E03 Inquiry Biochemistry** at McMaster University. The project demonstrates how machine learning can be applied to real-world cancer genomics data to understand chemotherapy response and resistance mechanisms.

### Learning Objectives

By exploring this application, students will:

1. **Apply Biochemical Concepts to Medicine:** Connect molecular mechanisms (DNA repair, cell signaling, apoptosis) to clinical outcomes in cancer treatment
2. **Understand Drug-Target Interactions:** See how specific mutations affect therapeutic efficacy across different drug classes
3. **Explore Resistance Mechanisms:** Investigate intrinsic vs. acquired resistance at the molecular level
4. **Analyze Real-World Data:** Work with actual clinical genomics data from 1,060 NSCLC patients
5. **Experience Personalized Medicine:** Discover how genomic profiling guides treatment selection
6. **Interpret Machine Learning Predictions:** Use explainable AI to understand which biochemical features drive clinical outcomes

### Interactive Demonstration Features

The application allows hands-on exploration of:
- **Drug Class Effects:** Compare platinum agents, taxanes, targeted inhibitors, and immunotherapies
- **Mutation Impact Analysis:** Toggle specific mutations (EGFR, KRAS, TP53, ALK) to see their effect on predicted outcomes
- **Treatment Sequencing:** Understand why first-line and later-line treatments have different response patterns
- **Feature Importance:** Use SHAP analysis to identify which genomic and clinical factors most influence predictions
- **Combination Therapy:** Explore synergistic effects of multi-drug regimens

This tool bridges biochemistry lecture content with clinical application, demonstrating how molecular understanding translates to patient care decisions.

---

## Overview

This project is an **interactive web application** that predicts chemotherapy response in non-small cell lung cancer (NSCLC) patients using machine learning models trained on real-world clinical and genomic data. The application helps users explore how different patient characteristics, tumor mutations, and drug selections affect predicted treatment outcomes, specifically focusing on **progression-free survival (PFS)** and **resistance development**.

By analyzing patterns from over 2,900 treatment records, the models learn to identify factors associated with **intrinsic resistance** (initial non-response) and **acquired resistance** (disease progression after initial response) to various chemotherapy classes.

---

## Dataset: AACR Project GENIE

### What is GENIE?

The **AACR Project GENIE** (Genomics Evidence Neoplasia Information Exchange) is an international data-sharing consortium that aggregates real-world clinical and genomic data from cancer patients across multiple institutions. For this project, we specifically used the **GENIE NSCLC 2.0-public** dataset, which focuses on non-small cell lung cancer patients.

### Dataset Characteristics

- **Total Treatment Records:** 2,934 chemotherapy treatment episodes
- **Unique Patients:** 1,060 individuals with NSCLC
- **Genomic Features:** 1,318 features including:
  - Mutation status for key genes (TP53, EGFR, KRAS, ALK, etc.)
  - Copy number alterations
  - Tumor mutational burden
  - Cancer stage and histology
- **Drug Library:** 81 unique chemotherapy agents spanning multiple classes:
  - Platinum-based agents (Cisplatin, Carboplatin)
  - Taxanes (Paclitaxel, Docetaxel)
  - Antimetabolites (Gemcitabine, Pemetrexed)
  - Targeted small-molecule inhibitors (Gefitinib, Erlotinib, Osimertinib)
  - Immunotherapies (Pembrolizumab, Nivolumab)
  - And many others organized by mechanism of action

### Why GENIE Data is Valuable

Real-world clinical data from GENIE captures the complexity and heterogeneity of cancer treatment that clinical trials often miss:

1. **Diverse Patient Populations:** Multiple treatment lines, varying disease stages, different mutation profiles
2. **Real-World Outcomes:** Actual progression-free survival data, not idealized clinical trial endpoints
3. **Treatment Combinations:** Realistic combination therapies used in clinical practice
4. **Longitudinal Data:** Tracks how resistance develops across multiple treatment lines

---

## Understanding Chemotherapy Resistance

### Types of Resistance

Cancer cells can develop resistance to chemotherapy through two main mechanisms:

#### 1. **Intrinsic Resistance** (Primary Resistance)
Cancer cells are **already resistant** before treatment begins, due to:
- Pre-existing mutations that inactivate drug targets
- High expression of drug efflux pumps
- Alternative survival pathways
- Tumor microenvironment factors

**Example:** A patient with an *EGFR* mutation may have intrinsic resistance to platinum-based chemotherapy but respond well to EGFR-targeted therapies like Gefitinib.

#### 2. **Acquired Resistance** (Secondary Resistance)
Cancer initially responds to treatment but later develops resistance through:
- New mutations that bypass the drug's mechanism
- Amplification of alternative signaling pathways
- Epithelial-to-mesenchymal transition (EMT)
- Selection pressure favoring resistant clones

**Example:** A tumor initially sensitive to Gefitinib may acquire a *T790M* mutation in EGFR, requiring switch to a third-generation inhibitor like Osimertinib.

### How Drug Classes Differ in Resistance Patterns

Different chemotherapy classes have distinct mechanisms of action and therefore encounter different resistance mechanisms:

| **Drug Class** | **Mechanism** | **Common Resistance Mechanisms** |
|----------------|---------------|----------------------------------|
| **Platinum Agents** | DNA crosslinking | DNA repair pathway upregulation, nucleotide excision repair |
| **Taxanes** | Microtubule stabilization | β-tubulin mutations, drug efflux pumps (P-glycoprotein) |
| **EGFR Inhibitors** | Tyrosine kinase inhibition | T790M mutation, MET amplification, PIK3CA mutations |
| **Immunotherapies** | Immune checkpoint blockade | Loss of antigen presentation, immunosuppressive microenvironment |

---

## Machine Learning Models

This project uses a **stratified modeling approach** that recognizes treatment outcomes differ substantially between first-line and later-line therapies.

### Model 1: XGBoost (First-Line Treatment)

**Used for:** Patients receiving their **first chemotherapy regimen** (treatment line = 1)

- **Training Data:** 822 first-line treatment records
- **Algorithm:** Gradient Boosted Trees (XGBoost)
- **Feature Selection:** 500 most predictive features (from 9,510 total)
- **Performance:** AUROC = 0.703 for predicting 6-month clinical benefit
- **Why XGBoost?** Smaller dataset requires simpler model with aggressive feature selection to avoid overfitting

**What It Learns About Resistance:**
- Identifies genomic markers associated with intrinsic resistance (e.g., specific mutation combinations)
- Recognizes which drug classes work best for treatment-naive tumors
- Captures baseline tumor characteristics that predict initial response

### Model 2: Deep Learning (Previous Treatment)

**Used for:** Patients who have **received prior chemotherapy** (treatment line ≥ 2)

- **Training Data:** 2,112 records from second-line and beyond
- **Architecture:** Multi-modal neural network with cross-attention
- **Parameters:** ~23 million trainable parameters
- **Performance:** AUROC ≈ 0.70-0.75 (expected)
- **Why Deep Learning?** Larger dataset allows complex model to capture non-linear drug-genomic interactions

**Architecture Components:**
1. **Genomic Encoder:** Processes 1,318 patient/tumor features
2. **Drug Fingerprint Encoder:** Analyzes molecular structure of up to 4 drugs (8,192-dimensional fingerprints)
3. **Bidirectional Cross-Attention:** Learns how specific genomic features interact with drug mechanisms
4. **Output Head:** Predicts progression-free survival in continuous time

**What It Learns About Resistance:**
- Patterns of acquired resistance from previous treatments
- How prior drug exposure affects subsequent treatment response
- Complex interactions between mutation status and drug class selection
- Which combination therapies overcome resistance mechanisms

---

## Interactive Web Application

### How to Use the Predictor

The Streamlit-based web interface allows users to explore treatment predictions interactively:

#### **Step 1: Select Chemotherapy Regimen**
- Choose 1-4 drugs from 81 available agents
- Drugs organized by class for easy browsing:
  - Platinum Agents
  - Taxanes
  - EGFR/ALK Inhibitors
  - Immunotherapies
  - And 10+ other categories

#### **Step 2: Input Patient Characteristics**
- **Demographics:** Age, sex
- **Tumor Features:** Stage, histology
- **Genomic Profile:** Key mutations (TP53, KRAS, EGFR, ALK)
- **Treatment History:** Treatment line number, prior therapies

*Note: Fields left blank use dataset median values as defaults*

#### **Step 3: Generate Prediction**
The model predicts:
- **Progression-Free Survival (PFS):** Expected months until disease progression
- **Resistance Risk:** Probability of resistance development by 6 months
- **Confidence Score:** Model certainty in the prediction

#### **Step 4: Explore Feature Importance**
- See which factors most influence the prediction (using SHAP values)
- Understand whether mutations increase or decrease expected survival
- Identify genomic vulnerabilities that could be targeted

### Educational Value

This interactive demonstration helps users learn:

1. **Drug Class Effects:** How different chemotherapy mechanisms perform in various genomic contexts
2. **Mutation Impact:** Whether specific mutations confer resistance or sensitivity to certain drugs
3. **Treatment Sequencing:** Why later treatment lines often have worse outcomes (acquired resistance)
4. **Combination Therapy:** How multi-drug regimens can overcome single-agent resistance
5. **Personalized Medicine:** The importance of matching treatment to tumor genomic profile

### Example Scenarios for HTHSCI 2E03 Students

**Scenario 1: EGFR-Mutant NSCLC (Targeted Therapy Concept)**
- **Question:** How does an EGFR mutation affect response to targeted vs. traditional chemotherapy?
- **Method:** Try Gefitinib (EGFR inhibitor) with EGFR mutation present vs. absent
- **Expected Learning:** Understand how driver mutations create targetable vulnerabilities
- **Biochemical Concept:** Tyrosine kinase inhibition disrupts aberrant signaling pathways

**Scenario 2: Treatment Line Effect (Acquired Resistance)**
- **Question:** Why do patients respond worse to later treatment lines?
- **Method:** Predict outcome for same patient at line 1 vs. line 3
- **Expected Learning:** Acquired resistance accumulates through clonal selection
- **Biochemical Concept:** Tumor evolution selects for resistant phenotypes

**Scenario 3: TP53 Mutation Impact (Apoptosis Defects)**
- **Question:** How does loss of p53 tumor suppressor affect chemotherapy response?
- **Method:** Compare predictions with TP53 mutation present vs. absent
- **Expected Learning:** Apoptosis-inducing drugs require functional p53 pathway
- **Biochemical Concept:** p53 mediates DNA damage response and programmed cell death

**Scenario 4: Combination vs. Monotherapy (Synergy)**
- **Question:** Why are combination regimens often more effective?
- **Method:** Compare single-agent platinum vs. platinum + taxane combination
- **Expected Learning:** Multi-target approaches reduce resistance probability
- **Biochemical Concept:** Attacking multiple cellular processes simultaneously

---

## Technical Implementation

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (if needed)
python train_xgboost.py          # First-line XGBoost model
python train_stratified.py       # Later-line Deep Learning model

# Launch web application
streamlit run demo/app.py
```

### Project Structure

```
cancerchemo/
├── demo/
│   └── app.py                   # Streamlit web application
├── notebooks/
│   └── model.py                 # Deep learning architecture
├── checkpoints_stratified/
│   ├── first_treatment_xgboost/ # XGBoost model files
│   └── previous_treatment/      # Deep learning checkpoints
├── artifacts/
│   ├── feature_names.json       # Feature list
│   ├── drug_fp_library.npz      # Pre-computed drug fingerprints
│   └── sample_metadata.csv      # Training data statistics
├── data/
│   └── processed/
│       ├── drug_smiles_cache.json
│       └── drug_classes.json
├── train_xgboost.py             # Train first-line model
└── train_stratified.py          # Train later-line model
```

### Key Technologies

- **PyTorch:** Deep learning framework for neural network models
- **XGBoost:** Gradient boosting for tabular data
- **RDKit:** Chemical informatics for drug fingerprint generation
- **SHAP:** Explainable AI for feature importance
- **Streamlit:** Interactive web application framework
- **Plotly:** Interactive visualization

---

## Clinical Interpretation

### Model Outputs Explained

**Progression-Free Survival (PFS):**
- **≥6 months:** Likely responder (clinical benefit)
- **3-6 months:** Intermediate response
- **<3 months:** Likely poor response (intrinsic resistance)

**Resistance Risk:**
- **<30%:** Low risk (favorable prognosis)
- **30-60%:** Moderate risk (monitor closely)
- **>60%:** High risk (consider alternative therapy)

**Confidence Score:**
- **0.7-1.0:** High confidence (prediction far from decision boundary)
- **0.4-0.7:** Moderate confidence
- **<0.4:** Low confidence (close to 6-month threshold)

### Important Limitations

⚠️ **This is a research and educational tool, NOT for clinical decision-making.**

- Models trained on historical data may not generalize to all populations
- Does not account for patient comorbidities, quality of life, or treatment compliance
- Drug interactions and dose modifications not modeled
- Genomic data may be incomplete (not all genes sequenced in GENIE)
- **Always consult with oncologists for actual treatment decisions**

---

## Learning Objectives

By exploring this application, students and researchers can:

1. **Understand Real-World Cancer Data:** See how clinical genomics data is structured and used
2. **Explore Machine Learning in Healthcare:** Learn how models are trained on medical data
3. **Investigate Resistance Mechanisms:** Discover how mutations affect drug response
4. **Compare Drug Classes:** Understand differences between chemotherapy mechanisms
5. **Appreciate Model Stratification:** Learn why different clinical scenarios need different models
6. **Practice Explainable AI:** Use SHAP values to interpret black-box predictions

---

## References

**Dataset:**
> The AACR Project GENIE Consortium. (2017). AACR Project GENIE: Powering Precision Medicine Through An International Consortium. *Cancer Discovery*, 7(8), 818-831. [https://doi.org/10.1158/2159-8290.CD-17-0151](https://doi.org/10.1158/2159-8290.CD-17-0151)

**GENIE Data Version:** NSCLC 2.0-public

**Key Papers on Chemotherapy Resistance:**
- Holohan, C., et al. (2013). Cancer drug resistance: an evolving paradigm. *Nature Reviews Cancer*, 13(10), 714-726.
- Housman, G., et al. (2014). Drug resistance in cancer: an overview. *Cancers*, 6(3), 1769-1792.

---

## Course Information

**Course:** HTHSCI 2E03 Inquiry Biochemistry  
**Institution:** McMaster University  
**Project Focus:** Application of biochemical principles to personalized cancer medicine

This project demonstrates the integration of:
- Molecular biology (mutation effects on protein function)
- Biochemistry (drug-target interactions, metabolic pathways)
- Pharmacology (drug mechanisms and resistance)
- Data science (machine learning on clinical data)
- Clinical medicine (treatment outcomes and decision-making)

---

## Contributors

This project was developed as an educational demonstration for HTHSCI 2E03 at McMaster University, showcasing machine learning applications in precision oncology using publicly available GENIE consortium data.

---

## License

This project is for educational and research purposes. GENIE data usage follows AACR Project GENIE data use guidelines. Models and code are provided as-is for non-commercial educational use.

---

## Future Directions

Potential enhancements for this demonstration:

- **Toxicity Prediction:** Model side effects alongside efficacy
- **Survival Curves:** Full Kaplan-Meier curve prediction
- **Multi-Cancer Support:** Extend beyond NSCLC to other tumor types
- **Longitudinal Tracking:** Predict resistance evolution over multiple lines
- **Biomarker Discovery:** Identify novel resistance-associated mutations
- **Drug Combination Optimization:** Recommend optimal multi-drug regimens

---

**Questions or Issues?** This is a demonstration project for educational purposes. For questions about the GENIE dataset, visit [www.aacr.org/professionals/research/aacr-project-genie/](https://www.aacr.org/professionals/research/aacr-project-genie/)
