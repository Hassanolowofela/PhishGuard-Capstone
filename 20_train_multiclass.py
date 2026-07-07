#!/usr/bin/env python3
"""Phase 4 extension: train the Safe/Scam/Malware multiclass model.

Single-source (Zenodo multiclass NLP dataset) to avoid the source artifact that a
CEAS merge would introduce. Reports honest, leakage-controlled (near-duplicate
group-aware) per-class metrics, then fits the final model on all data and saves it
for the web app.

Usage:  python 20_train_multiclass.py [data_csv] [models_dir]
Defaults: data/phishing_nlp_dataset.csv  and  models/
Outputs: models/20_multiclass_model.joblib, models/20_multiclass_vectorizer.joblib,
         data/20_multiclass_report.json
"""
import os, sys, json, re, unicodedata
import numpy as np, pandas as pd, scipy.sparse as sp, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "phishing_nlp_dataset.csv")
MODELS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "models")
DATADIR = os.path.join(HERE, "data"); os.makedirs(MODELS, exist_ok=True); os.makedirs(DATADIR, exist_ok=True)
SEED = 42

KNOWN = {"Phishing","Malware","Scareware","Baiting","Pretexting","NOT-Malicious General Class","NOT-Malicious"}
MAP = {"NOT-Malicious General Class":"safe","NOT-Malicious":"safe","Phishing":"scam",
       "Scareware":"scam","Baiting":"scam","Pretexting":"scam","Malware":"malware"}
LABELS = ["safe","scam","malware"]

# ---- feature logic (identical to webapp/features.py) ----
URL_RE=re.compile(r"https?://\S+|www\.\S+",re.I); EMAIL_RE=re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS_RE=re.compile(r"\s+"); NONPRINT=re.compile(r"[^\x20-\x7E]"); HTML_RE=re.compile(r"<[^>]+>"); WORD=re.compile(r"\b\w+\b")
URGENT=["urgent","verify","suspend","account","password","click","login","update","confirm","winner","won","prize","free","limited","act now","security","alert","bank","ssn","invoice","payment","refund"]
STRUCT_ORDER=["body_char_len","body_word_count","subject_char_len","subject_word_count","num_urls","has_url","num_html_tags","has_html","num_exclaim","num_question","num_digits","digit_ratio","uppercase_ratio","num_money_symbols","has_money_symbol","avg_word_len","urgent_word_count","subject_is_reply"]
def clean_text(raw):
    t=unicodedata.normalize("NFKD",str(raw)); t=NONPRINT.sub(" ",t); t=URL_RE.sub(" urltoken ",t); t=EMAIL_RE.sub(" emailtoken ",t)
    t=t.lower(); t=re.sub(r"[^a-z0-9\s]"," ",t); return WS_RE.sub(" ",t).strip()
def structural(subject, body):
    s,b=str(subject),str(body); text=s+" "+b; low=text.lower(); w=WORD.findall(text); nw=max(len(w),1); L=[c for c in text if c.isalpha()]
    d={"body_char_len":len(b),"body_word_count":len(WORD.findall(b)),"subject_char_len":len(s),"subject_word_count":len(WORD.findall(s)),
       "num_urls":len(URL_RE.findall(text)),"has_url":int(bool(URL_RE.search(text))),"num_html_tags":len(HTML_RE.findall(b)),
       "has_html":int(bool(HTML_RE.search(b))),"num_exclaim":text.count("!"),"num_question":text.count("?"),
       "num_digits":sum(c.isdigit() for c in text),"digit_ratio":round(sum(c.isdigit() for c in text)/max(len(text),1),4),
       "uppercase_ratio":round(sum(c.isupper() for c in L)/max(len(L),1),4),"num_money_symbols":sum(text.count(c) for c in "$£€"),
       "has_money_symbol":int(any(c in text for c in "$£€")),"avg_word_len":round(sum(len(x) for x in w)/nw,3),
       "urgent_word_count":sum(low.count(u) for u in URGENT),"subject_is_reply":int(s.strip().lower().startswith(("re:","fw:","fwd:")))}
    return [d[k] for k in STRUCT_ORDER]

def parse(cell):
    s=str(cell)
    if "\t" in s:
        pre,post=s.rsplit("\t",1); post=post.strip()
        if post in KNOWN: return pre.strip(),post
    for k in sorted(KNOWN,key=len,reverse=True):
        if s.strip().endswith(k): return s.strip()[:-len(k)].strip(),k
    return s.strip(),None

# ---- load + map + clean + dedupe ----
df=pd.read_csv(DATA); pr=df["Corpus"].map(parse); df["text"]=[p[0] for p in pr]; df["rl"]=[p[1] for p in pr]
df=df[df["rl"].notna()].copy(); df["y"]=df["rl"].map(MAP)
df["clean"]=df["text"].map(lambda b: clean_text(" . "+str(b)))
df=df[df["clean"].str.len()>0].drop_duplicates("clean").reset_index(drop=True)
y=df["y"].values; n=len(df)
Xs=np.array([structural("", b) for b in df["text"]], dtype=np.float64)

# ---- near-duplicate groups (unsupervised) for a leakage-controlled estimate ----
Vg=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=1).fit(df["clean"])
Mg=normalize(Vg.transform(df["clean"])); Sg=(Mg@Mg.T); Sg.setdiag(0); Sg.data=(Sg.data>=0.9).astype(int); Sg.eliminate_zeros()
_,groups=connected_components(Sg,directed=False)

def build(idx, vec, fit):
    ct=df["clean"].iloc[idx]
    Xt=vec.fit_transform(ct) if fit else vec.transform(ct)
    return sp.hstack([sp.csr_matrix(Xs[idx]),Xt]).tocsr()

# honest group-aware held-out estimate
sg=StratifiedGroupKFold(4,shuffle=True,random_state=SEED); tr,te=next(sg.split(np.arange(n),y,groups=groups))
vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_df=0.9,max_features=2000,sublinear_tf=True)
Xtr=build(tr,vec,True); Xte=build(te,vec,False)
gs=GridSearchCV(make_pipeline(MaxAbsScaler(),LogisticRegression(max_iter=4000,class_weight="balanced")),
                {"logisticregression__C":[1,3,10]},scoring="f1_macro",cv=StratifiedKFold(4,shuffle=True,random_state=SEED),n_jobs=-1)
gs.fit(Xtr,y[tr]); yp=gs.predict(Xte)
rep=classification_report(y[te],yp,digits=3,labels=LABELS,output_dict=True)
cm=confusion_matrix(y[te],yp,labels=LABELS).tolist()
print("Honest group-aware held-out per-class F1:",{k:round(rep[k]["f1-score"],3) for k in LABELS},"macro",round(rep["macro avg"]["f1-score"],3))

# ---- final model: fit vectorizer + classifier on ALL data ----
final_vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_df=0.9,max_features=2000,sublinear_tf=True)
Xt_all=final_vec.fit_transform(df["clean"]); X_all=sp.hstack([sp.csr_matrix(Xs),Xt_all]).tocsr()
bestC=gs.best_params_["logisticregression__C"]
final=make_pipeline(MaxAbsScaler(),LogisticRegression(max_iter=6000,class_weight="balanced",C=bestC)).fit(X_all,y)
joblib.dump(final,os.path.join(MODELS,"20_multiclass_model.joblib"))
joblib.dump(final_vec,os.path.join(MODELS,"20_multiclass_vectorizer.joblib"))
report={"n_messages":int(n),"class_counts":df["y"].value_counts().to_dict(),"best_C":bestC,
        "honest_group_aware_test":{"per_class":{k:{m:round(rep[k][m],3) for m in ["precision","recall","f1-score","support"]} for k in LABELS},
        "macro_f1":round(rep["macro avg"]["f1-score"],3),"accuracy":round(rep["accuracy"],3),"test_n":int(len(te)),
        "confusion_rows_true":cm,"labels":LABELS},
        "note":"Text-based detection of malware-LURE emails, not payload scanning. Malware class is small (proof of concept)."}
json.dump(report,open(os.path.join(DATADIR,"20_multiclass_report.json"),"w"),indent=2)
print("Saved model + vectorizer to",MODELS,"| classes:",list(final.classes_))
