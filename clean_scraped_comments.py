import pandas as pd
import re

# Load data
df = pd.read_csv("komentar_ig_raw.csv")

# Pastikan kolom komentar berupa string
df['comment'] = df['comment'].astype(str)

# 1. Hapus emoji
emoji_pattern = re.compile(
    "["u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE)
df['comment'] = df['comment'].apply(lambda x: emoji_pattern.sub(r'', x))

# 2. Hapus mention (@username)
df['comment'] = df['comment'].str.replace(r'@\w+', '', regex=True)

# 3. Hapus hashtag (#tag)
df['comment'] = df['comment'].str.replace(r'#\w+', '', regex=True)

# 4. Hapus URL
df['comment'] = df['comment'].str.replace(r'http\S+|www.\S+', '', regex=True)

# 5. Hapus karakter berulang dan simbol tidak penting
df['comment'] = df['comment'].str.replace(r'[^A-Za-z0-9\s.,]', ' ', regex=True)
df['comment'] = df['comment'].str.replace(r'\s+', ' ', regex=True)

# 6. Lowercase
df['comment'] = df['comment'].str.lower().str.strip()

# 7. Hapus komentar kosong setelah cleaning
df = df[df['comment'].str.len() > 1]

# 8. Opsi tambahan — filter komentar SPAM
spam_words = [
    "follow", "cek bio", "open order", "dm", "giveaway",
    "promo", "diskon", "langsung wa", "jasa", "jual"
]
pattern_spam = '|'.join(spam_words)
df = df[~df['comment'].str.contains(pattern_spam, na=False)]

# Simpan hasil cleaning
df.to_csv("komentar_ig_clean.csv", index=False)

print("Cleaning komentar Instagram selesai! Hasil: komentar_ig_clean.csv")