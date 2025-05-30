# src/utils/tokenization.py
VOCAB = ["acqua", "cibo", "Sonno", "aiuto", "grazie", "per favore", "ciao", "addio", "sì", "no", "questo", "quello", "bene", "questa", "mamma", "papà", "me", "io", "tu", "dare", "rosso", "blu", "verde"]
TOKEN2ID = {w: i for i, w in enumerate(VOCAB)}
ID2TOKEN = {i: w for w, i in TOKEN2ID.items()}
