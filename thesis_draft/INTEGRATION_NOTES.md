# Integrating the new sections

The editable draft is `missing_sections.docx`; its source is `missing_sections.md`.
Insert the existing literature review as Chapter 2 between the Introduction and
Research Questions. The supplied literature-review file has not been modified.
The suggested title is provisional and can be replaced with the registered thesis title.

## Model identity and experiment scope

The repository's earlier CNN is **EfficientNet-B4**, not ResNet. The draft uses
the verified name. ResNet appears in the literature review as another study's
architecture, not as an implementation with results in this repository.

EfficientNet is the primary convolutional reference. Both DINOv2 variants are
reported. The five-fold results form the main experiment; the historical 73.7%
pilot accuracy is described separately and is not substituted for the current
58.33% EfficientNet estimate. Keeping EfficientNet as the primary approach does
not imply it dominated all metrics or that DINOv2 was disproved in general.

## Details to settle before submission

1. Confirm the professor's preferred name/role, annotation instructions, whether
   numerical metadata were consulted, and how Stage III/IV were distinguished.
   The draft states the current documentation limits rather than inventing a
   clinical adjudication procedure or assuming an images-only annotation process.
2. Clarify why all ten Stage I labels correspond to BRAR bone-resorption values
   above 15%. Do not silently relabel them. If labels change, the current results
   describe the original labels and must not be represented as evaluation of the
   revised labels.
3. Confirm whether one image identifier always denotes one patient. The current
   folds assume this; exact-image duplicate checks cannot establish it.
4. Confirm the institution's wording and requirements for secondary use of the
   dataset. No new ethics approval number, waiver, or consent claim was invented.
5. Reconcile the source publication's 1,104 participants with the 988 locally
   available image records. The local analysis did not document 116 exclusions.

## Align the existing literature review with the completed study

- Correct the fractional/percentage convention in the BRAR formula. The actual
  CSV satisfies `Bone resorption Age = 100 * Bone resorption / Age` to rounding.
  A fraction such as 0.30 corresponds to 30%, not 0.30%, when applying
  percentage-based thresholds. Calling the age-normalized ratio dimensionless
  also requires care because the denominator is age in years.
- The experiments predict stage labels only. Modify language implying that this
  thesis has validated both staging and grading; grading remains background.
- BRAR's original levels 1/2/3 are not Stage I/II/III. The current four-stage
  subset happens to lie entirely in original level_3.
- The actual local archive contains JPGs and one metadata CSV, not CEJ/apex
  coordinates or segmentation masks. Avoid implying that such BRAR assets were
  used in this work. Regional crops are geometric, not detected teeth.
- Replace draft-facing language such as “For our paper ... this literature
  review has to show” with final academic prose.
- Reconcile the review's Dujic/Hoss naming for the 21,819-image study and the
  Al Husaini/Hasani attribution around Figure 2.1 against the original sources.
  These discrepancies were noticed in the supplied text; this is not a full
  verification of every reference or numerical claim in the review.
- Merge the four references in the new draft with the existing bibliography.
  Tonetti's framework was published jointly in two journals; use one consistent
  version. The draft cites the Journal of Periodontology version. Xia's record
  has a December 2025 online publication and a later volume assignment; use the
  institution's required bibliographic convention consistently.
- The exploratory RQs reflect the study rationale, not a claim of preregistration.
  No p-values or formal equivalence conclusions have been manufactured.

## Diagrams

Three original methodology diagrams are embedded in the Word draft. Each is also
available in `figures/` as PNG (300 dpi), SVG (editable vector), and PDF (vector).
Editable diagram logic is provided in the corresponding `.mmd` Mermaid files.
The exported graphics were drawn programmatically by `build_documents.py`.

## Rebuilding

From the repository root, using the existing Python environment:

```powershell
python thesis_draft/build_documents.py
```

The local document-export dependency is installed under `.build_deps/` and is
excluded from version control. On another machine install `python-docx` and
`matplotlib` into the environment or install python-docx with
`python -m pip install --target thesis_draft/.build_deps python-docx`.

Experiment evidence: `small_data_pipeline/artifacts/evaluation/` and
`small_data_pipeline/artifacts/checks/trained_checkpoints.json`.
