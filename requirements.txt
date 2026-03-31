# Profit Mix Optimizer

A Streamlit app for optimizing investment fund mixes across savings tracks —
supporting קרנות השתלמות, פוליסות חיסכון, קרנות פנסיה, קופות גמל, and גמל להשקעה.

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Notes

- **Google Sheets access** — product data is loaded from Google Sheets.  
  Each sheet must be shared publicly (*Anyone with the link – Viewer*).

- **Streamlit Secrets** — the following secrets must be configured in  
  `Settings → Secrets` on Streamlit Cloud (or in `.streamlit/secrets.toml` locally):

  | Secret | Purpose |
  |---|---|
  | `APP_PASSWORD` | Login password for the app |
  | `[gcp_service_account]` | GCP service account JSON for voting/write access |
  | `ANTHROPIC_API_KEY` | *(Optional)* Enables AI explanations on result cards |

  See `SETUP_VOTING.md` for step-by-step instructions on configuring the  
  service account and enabling community voting stats.
