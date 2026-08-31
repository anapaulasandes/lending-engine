# Credit Expansion Decisioning

A decision-support prototype for staged credit-market expansion. The application turns comparable-market evidence, business constraints, unit economics, and observed repayment outcomes into an auditable initial lending policy.

The current demonstration models an unsecured installment-loan launch in Panama using **Lumeria (fictional)** as the single synthetic comparable market.

> **Illustrative case study.** All portfolio assumptions, risk parameters, business constraints, and numerical results are synthetic and do not represent proprietary company data. Public market information is used only for contextual illustration.

## What It Does

The Streamlit dashboard follows a four-step decision flow:

1. **Understand the market**: capture the target population, evidence basis, and transferability assumptions.
2. **Define business constraints**: set the loss budget, portfolio exposure, pilot capacity, ticket ceiling, and maximum term.
3. **Recommend an initial policy**: select the first cohort, ticket, term, and deployment gate with explicit rules and constraints.
4. **Learn and adapt**: update risk expectations after mature outcomes and apply a reproducible policy-transition rule.

The main view is intentionally concise. Explanation toggles expose formulas, candidate comparisons, decision rules, diagnostic charts, sources, and numeric reproduction traces for review by a Credit Data Scientist.

## Auditability

Each recommendation can be inspected to answer:

- Which inputs and constraints were used?
- Which candidates were evaluated?
- Which formula produced each metric?
- Why did the selected candidate qualify?
- How can the calculation be reproduced in Python, SQL, or a notebook?

The default MVP assumptions are:

| Parameter | Default |
| --- | ---: |
| Prior expected PD | 10% |
| Prior strength | 20 equivalent observations |
| Loss given default | 70% |
| Initial ticket | $200 |
| Fee rate | 20% |
| Initial cohort | 10 customers |
| Maximum acceptable learning loss | $4,000 |
| Maximum portfolio exposure | $30,000 |

Expected Unit Economics is calculated consistently as:

$$
	ext{Expected UE} = \text{Revenue} - \text{Expected Credit Loss} - \text{Funding Cost} - \text{Operating Cost} - \text{Expected Collection Cost}
$$

where:

$$
	ext{Expected Credit Loss} = \text{Loan Amount} \times \text{PD} \times \text{LGD}
$$

$$
	ext{Expected Collection Cost} = \text{PD} \times \text{Collection Cost per Default}
$$

All demo values are illustrative and are not credit-policy advice.

## Repository Layout

```text
dashboard/expansion_demo/app.py  Streamlit expansion decisioning dashboard
engine/                          Lending-engine calculations and decision logic
tests/                           Automated tests
examples/sample_data/            Illustrative inputs and outputs
docs/                            GitHub Pages documentation site
```

## Run Locally

Use Python 3.12 or later. Create and activate a virtual environment, then install the dashboard dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy
streamlit run dashboard/expansion_demo/app.py
```

Open `http://localhost:8501` in a browser.

To run the test suite:

```bash
pytest
```

## GitHub Pages

GitHub Pages hosts the static documentation site in `docs/`. The workflow at [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) publishes it automatically after a push to `main`.

Before the first deployment:

1. Push this repository to GitHub.
2. In **Settings > Pages**, select **GitHub Actions** as the source.
3. Push to `main`, or run **Deploy GitHub Pages** from the Actions tab.
4. Open `https://<owner>.github.io/<repository>/`.

The Streamlit dashboard itself requires a Python runtime and therefore cannot run on GitHub Pages. Deploy it to a Python-capable host such as Streamlit Community Cloud, Render, or Azure App Service, then add its public URL to the documentation page.

## Development Notes

- Keep primary recommendations business-focused; technical methods belong in explanation toggles.
- Maintain a single source of truth for assumptions and derived values.
- Do not hardcode a displayed recommendation when it can be calculated from the displayed inputs.
- Preserve USD for the Panama demonstration.
