# FaithTrace demo corpus

Ten small, openly licensed, **non-government** documents (about 1.8 MB in total). They exercise the two things FaithTrace measures:

- **temporal drift**: two license families, each in two dated versions whose terms differ;
- **multimodal retrieval**: real tables, charts whose labels exist only in the image, an XLSX workbook and a CSV export.

`manifest.json` holds the machine-readable metadata for each file: source URL pinned to a commit, license, version label, effective dates, description and SHA-256. `scripts/seed.py` reads it to upload the corpus. `eval_sets/faithtrace_v1.json` holds the questions.

Every file is an unmodified copy of the upstream file. Only the filename was changed.

## Files

| File | Source | License | Version / dates | Why it is included |
|---|---|---|---|---|
| `cc_by_3.0_legalcode.html` | [creativecommons/cc-legal-tools-data](https://github.com/creativecommons/cc-legal-tools-data/blob/a0dac18d1a773b4cdb7d8dd5d9bf4efab7901030/docs/licenses/by/3.0/legalcode.en.html) | CC0 1.0 | 3.0, 2007-02-23 → 2013-11-24 | Older half of **versioned pair 1** (CC BY). It has no cure period, a different survival clause, a different liability exclusion and no database-rights section. |
| `cc_by_4.0_legalcode.html` | same repo, `docs/licenses/by/4.0/legalcode.en.html` | CC0 1.0 | 4.0, 2013-11-25 → current | Newer half of pair 1. It adds 30-day reinstatement, Section 4 Sui Generis Database Rights and a "direct" damages exclusion. |
| `cc_by_sa_3.0_legalcode.html` | same repo, `docs/licenses/by-sa/3.0/legalcode.en.html` | CC0 1.0 | 3.0, 2007-02-23 → 2013-11-24 | Older half of **versioned pair 2** (CC BY-SA). Adaptations may only use this license, a later version, or a CC jurisdiction port. |
| `cc_by_sa_4.0_legalcode.html` | same repo, `docs/licenses/by-sa/4.0/legalcode.en.html` | CC0 1.0 | 4.0, 2013-11-25 → current | Newer half of pair 2. It adds "BY-SA Compatible License". |
| `cncf_ambassador_midyear_survey_2023.pdf` | [cncf/surveys](https://github.com/cncf/surveys/blob/e2dd7c5405be150184d8f6921340927fccac7db2/ambassador/2023/CNCF_Ambassador_MidYear_Survey_2023.pdf) | Apache-2.0 | 2023-midyear (deck dated 11.29.2023) | **Chart document.** Pages 4–8 are pie charts. The question titles and category labels exist only in the chart images, and the PDF text layer holds just the bare percentages. Page 17 has a small initiatives table. |
| `cncf_toc_survey_2020_h1_matrix.pdf` | cncf/surveys, `toc/h1/Matrix-h1.pdf` | Apache-2.0 | 2020-H1 | **Table document.** Pages 1–3 are bar charts with no values in text. Page 4 is a results table with percentages, counts and weighted averages. |
| `cncf_2022_survey_reference_model.pdf` | cncf/surveys, `cloudnative/CNCF 2022 Survey Reference Model.pdf` | Apache-2.0; content states CDLA-Permissive-2.0 | 2022 | **Table document.** A survey instrument laid out as page / Q# / R# / response / show-hide-disqualify logic columns. |
| `cncf_2022_survey_csv_guide.pdf` | cncf/surveys, `cloudnative/2022 CNCF Cloud Native Survey - Guide to Using the CSV Files and Data.pdf` | Apache-2.0; content states CDLA-Permissive-2.0 | 2022 | Prose on how the data is encoded, plus a keep/filter-out criteria table. |
| `cncf_cloud_native_survey_2021_part1.xlsx` | cncf/surveys, `cloudnative/Cloud_Native_Survey_2021-Part_1.xlsx` | Apache-2.0 | 2021-part1 | **Spreadsheet.** 27 sheets (one per question) cross-tabulated by region, with paired percent and count columns and Total/Answered/Skipped rows. |
| `cncf_maintainers_survey_2020_h2_summary.csv` | cncf/surveys, `maintainer/2020/h2/CNCF_Maintainers_Survey_2020_H2_Summary.csv` | Apache-2.0 | 2020-H2 | **CSV.** Per-question blocks with answer counts, 1–10 rating distributions and weighted averages. It holds aggregates only, with no names or emails. |

## Licenses

- **Creative Commons legal code.** The `creativecommons/cc-legal-tools-data` repository is released under CC0 1.0 (see its `COPYING`). Each legal-code page also says: *"The text of the Creative Commons public licenses is dedicated to the public domain under the CC0 Public Domain Dedication."*
- **CNCF survey files.** The `cncf/surveys` repository is licensed under the Apache License 2.0 (see its `LICENSE`). The 2022 survey guide also points to the Community Data License Agreement – Permissive 2.0 (https://cdla.dev/permissive-2-0/) for that content. Both licenses allow redistribution. The files are © Cloud Native Computing Foundation and are included unmodified.

No government sources were used. Every file was fetched from public GitHub repositories with `git clone`.

## Effective dates

`effective_from` / `effective_to` give the period during which a version was the **current** one. FaithTrace uses these dates to choose the version that was valid at the query date.

- **CC 3.0 / 4.0.** The dates are the Creative Commons release dates: 3.0 launched on 2007-02-23 ([CC blog](https://creativecommons.org/2007/02/23/version-30-launched/)) and 4.0 launched on 2013-11-25 ([CC blog](https://creativecommons.org/2013/11/25/ccs-next-generation-licenses-welcome-version-4-0/)). 3.0 is treated as superseded the day before 4.0 launched. These dates do not appear in the HTML files themselves. The 3.0 page does say "This is an older version of this license" and points to 4.0.
- **Ambassador deck.** `effective_from` is the "Last Updated 11.29.2023" date printed on page 1.
- **Other CNCF files.** Both dates are `null` because the files carry no effective date. The half-year or year is kept in `version_label`.

## Known caveats

- The two versioned pairs are both Creative Commons licenses. Each pair is a genuinely different dated version of the same legal text, but the drift scenarios all come from one publisher.
- The CC 3.0 pages contain a notice that links to and names the 4.0 license. This is a realistic source of wrong-version retrieval.
- The 2022 reference-model PDF has the page header "2022 State of FinOps - Programming Reference", a template artifact present in the upstream file.
- The XLSX includes the respondents' free-text "Other (please specify)" answers exactly as published upstream.
