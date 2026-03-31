# Profit Mix Optimizer

כלי לאופטימיזציית תמהיל מסלולי חיסכון לסוכני ביטוח.

## מבנה הפרויקט

```
profit-mix-optimizer/
├── app.py                          ← נקודת כניסה ראשית (Streamlit)
├── loader.py                       ← טעינת נתונים מ-Google Sheets
├── optimizer.py                    ← מנוע חישוב תמהיל + אופטימיזציה
├── ui_components.py                ← כל רכיבי הממשק
├── streamlit_app.py                ← גרסה ישנה (לא בשימוש)
├── requirements.txt
├── runtime.txt
└── institutional_strategy_analysis/  ← מודול ניתוח היסטורי (לשלב ב')
```

## הרצה מקומית

```bash
pip install -r requirements.txt
streamlit run app.py
```

## פריסה ל-Streamlit Cloud

1. העלה את הפרויקט ל-GitHub (repository פרטי)
2. היכנס ל-[share.streamlit.io](https://share.streamlit.io)
3. חבר את ה-repository
4. הגדר Secrets (ראה למטה)
5. Main file path: `app.py`

## Streamlit Secrets

ב-Streamlit Cloud → Settings → Secrets:

```toml
APP_PASSWORD = "הסיסמה_שלך"
ANTHROPIC_API_KEY = "sk-ant-..."  # אופציונלי — להסברים AI

# אופציונלי — לסטטיסטיקות הצבעה
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."
```

## מקורות נתונים (Google Sheets)

כל ה-Google Sheets חייבים להיות פתוחים לצפייה ("Anyone with the link").

| מוצר | Sheet ID |
|------|----------|
| קרנות השתלמות | `1bO-Mdvz5sWw73J-5msGdeleK-F4csklG` |
| פוליסות חיסכון | `15XIxQxBU4Mfun-rgcwK0Mfq6blUgekRx` |
| קרנות פנסיה | `1EOBY0L2IVUyY8zBYBMKEBkBCixaD9v7m` |
| קופות גמל | `1JwKWUPj5TxMGbfQwABdZP0XIQOCvQHz9` |
| גמל להשקעה | `1gvA13OkDHBkf0QjJZQ_Z21jF8bNgAHgJ` |

## עדכון חודשי

מדי חודש — עדכני את קבצי ה-Google Sheets בנתונים החדשים.
האפליקציה שואבת את הנתונים באופן אוטומטי (cache של 15 דקות).
