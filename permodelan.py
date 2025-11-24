import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data jika belum ada
def download_nltk_resources():
    """Download required NLTK resources dengan error handling"""
    resources = {
        'vader_lexicon': 'sentiment/vader_lexicon',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
            print(f"✅ {resource} already available")
        except LookupError:
            try:
                print(f"📥 Downloading {resource}...")
                nltk.download(resource, quiet=True)
                print(f"✅ {resource} downloaded successfully")
            except Exception as e:
                print(f"⚠️  Could not download {resource}: {e}")

# Download resources
download_nltk_resources()

class InstagramCommentPreprocessor:
    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"⚠️  Error initializing NLP tools: {e}")
            self.setup_fallback_nlp()
        
        # Tambahan stopwords untuk media sosial
        self.social_media_stopwords = {
            'instagram', 'ig', 'post', 'photo', 'picture', 'pic', 'reel', 
            'story', 'feed', 'like', 'comment', 'share', 'follow', 'follower',
            'followers', 'following', 'dm', 'doubletap', 'tap', 'upload'
        }
        self.stop_words.update(self.social_media_stopwords)
    
    def setup_fallback_nlp(self):
        """Setup fallback jika NLP tools gagal"""
        print("🔄 Setting up fallback NLP tools...")
        self.sia = None
        self.lemmatizer = None
        self.stop_words = set()
    
    def load_and_clean_data(self, file_path):
        """Load dan bersihkan data komentar Instagram dengan error handling"""
        print(f"📥 Memuat data dari: {file_path}")
        
        try:
            # Cek jika file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} tidak ditemukan")
            
            # Load data dengan berbagai format
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except:
                    df = pd.read_csv(file_path, encoding='latin-1')
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                # Coba baca sebagai CSV dengan encoding berbeda
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except:
                    df = pd.read_csv(file_path, encoding='latin-1')
            
            # Basic info
            print(f"📊 Data loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Cari kolom yang mungkin berisi komentar
            comment_columns = ['comment', 'komentar', 'text', 'caption', 'content', 'comments']
            username_columns = ['username', 'user', 'author', 'pengguna']
            
            # Cari kolom komentar
            comment_col = None
            for col in comment_columns:
                if col in df.columns:
                    comment_col = col
                    break
            
            # Cari kolom username
            username_col = None
            for col in username_columns:
                if col in df.columns:
                    username_col = col
                    break
            
            if comment_col is None:
                # Jika tidak ada kolom yang sesuai, gunakan kolom pertama sebagai komentar
                comment_col = df.columns[0]
                print(f"⚠️  Kolom komentar tidak ditemukan, menggunakan kolom pertama: {comment_col}")
            
            if username_col is None:
                # Buat kolom username dummy
                df['username'] = ['user_' + str(i) for i in range(len(df))]
                username_col = 'username'
                print("⚠️  Kolom username tidak ditemukan, membuat username dummy")
            
            # Rename kolom untuk konsistensi
            df = df.rename(columns={comment_col: 'comment', username_col: 'username'})
            
            # Pastikan komentar berupa string dan handle NaN
            df['comment'] = df['comment'].astype(str)
            df['username'] = df['username'].astype(str)
            
            # Hapus baris dengan komentar kosong
            initial_count = len(df)
            df = df[df['comment'].str.strip() != '']
            df = df[df['comment'] != 'nan']
            final_count = len(df)
            
            if initial_count != final_count:
                print(f"✅ Komentar kosong dihapus: {initial_count - final_count}")
            
            print(f"📊 Data setelah cleaning awal: {len(df)} komentar")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def advanced_text_cleaning(self, df):
        """Pembersihan teks tingkat lanjut dengan error handling"""
        print("\n🧹 MEMULAI ADVANCED TEXT CLEANING...")
        
        try:
            # Simpan komentar original
            df['comment_original'] = df['comment'].copy()
            
            # 1. Hapus emoji
            emoji_pattern = re.compile(
                "["u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                "]+", flags=re.UNICODE)
            df['comment'] = df['comment'].apply(lambda x: emoji_pattern.sub(r'', str(x)))
            
            # 2. Hapus mention (@username)
            df['comment'] = df['comment'].str.replace(r'@\w+', '', regex=True)
            
            # 3. Hapus hashtag (#tag)
            df['comment'] = df['comment'].str.replace(r'#\w+', '', regex=True)
            
            # 4. Hapus URL
            df['comment'] = df['comment'].str.replace(r'http\S+|www\.\S+', '', regex=True)
            
            # 5. Hapus karakter khusus dan normalisasi
            df['comment'] = df['comment'].str.replace(r'[^A-Za-z0-9\s.,!?]', ' ', regex=True)
            df['comment'] = df['comment'].str.replace(r'\s+', ' ', regex=True)
            
            # 6. Convert to lowercase dan trim
            df['comment'] = df['comment'].str.lower().str.strip()
            
            # 7. Hapus komentar kosong setelah cleaning
            initial_count = len(df)
            df = df[df['comment'].str.len() > 2]
            final_count = len(df)
            print(f"✅ Komentar kosong dihapus: {initial_count - final_count}")
            
            print(f"📊 Setelah cleaning: {len(df)} komentar valid")
            return df
            
        except Exception as e:
            print(f"❌ Error dalam advanced text cleaning: {e}")
            return df
    
    def extract_text_features(self, df):
        """Ekstrak fitur-fitur dari teks komentar dengan error handling"""
        print("\n🔍 MENGEKSTRAK FITUR TEKS...")
        
        try:
            # 1. Basic text features
            df['comment_length'] = df['comment'].str.len()
            df['word_count'] = df['comment'].str.split().str.len()
            df['char_count'] = df['comment'].str.len()
            
            # 2. Advanced text features
            df['avg_word_length'] = df['comment'].apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
            )
            
            # 3. Punctuation features
            df['exclamation_count'] = df['comment'].str.count('!')
            df['question_count'] = df['comment'].str.count(r'\?')
            
            # 4. Social media specific features
            df['has_emoji_original'] = df['comment_original'].apply(self.contains_emoji)
            df['has_hashtag_original'] = df['comment_original'].str.contains(r'#\w+', na=False)
            df['has_mention_original'] = df['comment_original'].str.contains(r'@\w+', na=False)
            
            print("✅ Text feature extraction completed")
            return df
            
        except Exception as e:
            print(f"❌ Error dalam text feature extraction: {e}")
            return df
    
    def contains_emoji(self, text):
        """Cek apakah text mengandung emoji"""
        try:
            emoji_pattern = re.compile(
                "["u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                "]+", flags=re.UNICODE)
            return bool(emoji_pattern.search(str(text)))
        except:
            return False
    
    def perform_sentiment_analysis(self, df):
        """Analisis sentimen menggunakan VADER dengan fallback"""
        print("\n😊 MELAKUKAN ANALISIS SENTIMEN...")
        
        try:
            if self.sia is None:
                raise Exception("VADER sentiment analyzer tidak tersedia")
            
            # Analisis sentimen dengan VADER
            df['vader_scores'] = df['comment'].apply(self.sia.polarity_scores)
            df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
            df['vader_positive'] = df['vader_scores'].apply(lambda x: x['pos'])
            df['vader_negative'] = df['vader_scores'].apply(lambda x: x['neg'])
            df['vader_neutral'] = df['vader_scores'].apply(lambda x: x['neu'])
            
            # Kategorikan sentimen berdasarkan compound score
            conditions = [
                df['vader_compound'] >= 0.05,
                df['vader_compound'] <= -0.05,
                (df['vader_compound'] > -0.05) & (df['vader_compound'] < 0.05)
            ]
            choices = ['positive', 'negative', 'neutral']
            df['sentiment_vader'] = np.select(conditions, choices, default='neutral')
            
            print("✅ VADER sentiment analysis completed")
            
        except Exception as e:
            print(f"⚠️  VADER sentiment analysis failed: {e}")
            print("🔄 Using simple sentiment analysis instead...")
            # Set default values
            df['vader_compound'] = 0.0
            df['vader_positive'] = 0.0
            df['vader_negative'] = 0.0
            df['vader_neutral'] = 1.0
            df['sentiment_vader'] = 'neutral'
        
        # Sentimen sederhana berdasarkan kata kunci (fallback)
        df['sentiment_simple'] = df['comment'].apply(self.simple_sentiment_analysis)
        
        print("✅ Sentiment analysis completed")
        return df
    
    def simple_sentiment_analysis(self, text):
        """Analisis sentimen sederhana berdasarkan kata kunci"""
        try:
            positive_words = {
                'good', 'great', 'nice', 'love', 'awesome', 'amazing', 'beautiful', 
                'perfect', 'excellent', 'wonderful', 'fantastic', 'brilliant', 'best',
                'outstanding', 'superb', 'magnificent', 'lovely', 'gorgeous', 'stunning',
                'impressive', 'fabulous', 'terrific', 'happy', 'joy', 'pleased', 'delighted'
            }
            
            negative_words = {
                'bad', 'hate', 'terrible', 'awful', 'worst', 'dislike', 'ugly', 'horrible',
                'boring', 'disappointing', 'poor', 'weak', 'stupid', 'dumb', 'idiot',
                'sad', 'angry', 'mad', 'frustrated', 'annoyed', 'upset', 'hate'
            }
            
            text_lower = str(text).lower()
            words = set(text_lower.split())
            
            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def simple_tokenize(self, text):
        """Simple tokenization tanpa NLTK"""
        try:
            # Basic tokenization dengan split dan regex
            words = re.findall(r'\b[a-zA-Z]{2,}\b', str(text))  # Minimal 2 karakter
            return words
        except:
            return []
    
    def nlp_preprocessing(self, df):
        """Preprocessing untuk analisis NLP dengan fallback lengkap"""
        print("\n🔧 MEMPERSIAPKAN DATA UNTUK NLP...")
        
        try:
            # Tokenization dengan fallback
            try:
                from nltk.tokenize import word_tokenize
                df['tokens'] = df['comment'].apply(word_tokenize)
                print("✅ Using NLTK tokenizer")
            except Exception as e:
                print(f"⚠️  NLTK tokenizer failed, using simple tokenizer: {e}")
                df['tokens'] = df['comment'].apply(self.simple_tokenize)
            
            # Remove stopwords dan filtering
            if hasattr(self, 'stop_words') and self.stop_words:
                df['tokens_clean'] = df['tokens'].apply(
                    lambda tokens: [token for token in tokens 
                                  if token not in self.stop_words and len(token) > 2]
                )
            else:
                df['tokens_clean'] = df['tokens'].apply(
                    lambda tokens: [token for token in tokens if len(token) > 2]
                )
            
            # Reconstruct cleaned text untuk analisis
            df['comment_clean'] = df['tokens_clean'].apply(
                lambda tokens: ' '.join(tokens) if tokens else ''
            )
            
            print("✅ NLP preprocessing completed")
            return df
            
        except Exception as e:
            print(f"❌ Error dalam NLP preprocessing: {e}")
            # Fallback minimal
            df['tokens'] = df['comment'].apply(self.simple_tokenize)
            df['tokens_clean'] = df['tokens']
            df['comment_clean'] = df['comment']
            return df
    
    def user_behavior_analysis(self, df):
        """Analisis perilaku pengguna dengan error handling"""
        print("\n👤 ANALYZING USER BEHAVIOR...")
        
        try:
            # User statistics
            user_stats = df.groupby('username').agg({
                'comment': 'count',
                'comment_length': 'mean',
                'vader_compound': 'mean'
            }).round(3)
            
            user_stats = user_stats.rename(columns={
                'comment': 'comment_count',
                'comment_length': 'avg_comment_length',
                'vader_compound': 'avg_sentiment'
            })
            
            # Kategorikan user berdasarkan aktivitas
            user_stats['user_type'] = pd.cut(
                user_stats['comment_count'],
                bins=[0, 1, 3, float('inf')],
                labels=['casual', 'active', 'super_active']
            )
            
            print(f"📊 Total unique users: {len(user_stats)}")
            print(f"📈 User type distribution:\n{user_stats['user_type'].value_counts()}")
            
            return user_stats
            
        except Exception as e:
            print(f"❌ Error dalam user behavior analysis: {e}")
            # Return empty dataframe dengan struktur yang diharapkan
            return pd.DataFrame(columns=['comment_count', 'avg_comment_length', 'avg_sentiment', 'user_type'])
    
    def create_visualizations(self, df, user_stats):
        """Buat visualisasi data dengan error handling"""
        print("\n📊 MEMBUAT VISUALISASI...")
        
        try:
            # Buat folder untuk hasil
            if not os.path.exists('preprocessing_results'):
                os.makedirs('preprocessing_results')
            
            # Setup figure
            plt.figure(figsize=(15, 10))
            
            # 1. Sentiment distribution
            try:
                plt.subplot(2, 3, 1)
                sentiment_vader_counts = df['sentiment_vader'].value_counts()
                plt.pie(sentiment_vader_counts.values, labels=sentiment_vader_counts.index, 
                        autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
                plt.title('Sentiment Distribution (VADER)')
            except Exception as e:
                print(f"⚠️  Error creating sentiment pie chart: {e}")
                plt.subplot(2, 3, 1)
                plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                plt.title('Sentiment Distribution (VADER)')
            
            # 2. Comment length distribution
            try:
                plt.subplot(2, 3, 2)
                plt.hist(df['comment_length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Comment Length Distribution')
                plt.xlabel('Character Count')
                plt.ylabel('Frequency')
            except Exception as e:
                print(f"⚠️  Error creating comment length histogram: {e}")
                plt.subplot(2, 3, 2)
                plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                plt.title('Comment Length Distribution')
            
            # 3. Word count distribution
            try:
                plt.subplot(2, 3, 3)
                plt.hist(df['word_count'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.title('Word Count Distribution')
                plt.xlabel('Word Count')
                plt.ylabel('Frequency')
            except Exception as e:
                print(f"⚠️  Error creating word count histogram: {e}")
                plt.subplot(2, 3, 3)
                plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                plt.title('Word Count Distribution')
            
            plt.tight_layout()
            plt.savefig('preprocessing_results/basic_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Basic visualizations saved")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def save_processed_data(self, df, user_stats, filename_prefix="instagram_processed"):
        """Simpan data yang sudah diproses dengan error handling"""
        print("\n💾 MENYIMPAN DATA YANG SUDAH DIPROSES...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Buat folder jika belum ada
            if not os.path.exists('preprocessing_results'):
                os.makedirs('preprocessing_results')
            
            # 1. Save main processed data
            main_filename = f"preprocessing_results/{filename_prefix}_main_{timestamp}.csv"
            
            # Pilih kolom untuk output
            output_columns = [
                'username', 'comment_original', 'comment', 'comment_clean',
                'comment_length', 'word_count', 'sentiment_vader', 'sentiment_simple',
                'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral'
            ]
            
            # Hanya ambil kolom yang ada
            available_columns = [col for col in output_columns if col in df.columns]
            df[available_columns].to_csv(main_filename, index=False, encoding='utf-8')
            print(f"✅ Main data saved: {main_filename}")
            
            # 2. Save user statistics
            if not user_stats.empty:
                user_stats_filename = f"preprocessing_results/{filename_prefix}_user_stats_{timestamp}.csv"
                user_stats.to_csv(user_stats_filename, encoding='utf-8')
                print(f"✅ User statistics saved: {user_stats_filename}")
            else:
                user_stats_filename = None
                print("⚠️  User statistics is empty, skipping save")
            
            return main_filename, user_stats_filename
            
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
            return None, None
    
    def run_complete_preprocessing(self, file_path):
        """Jalankan preprocessing lengkap dengan comprehensive error handling"""
        print("🚀 MEMULAI PREPROCESSING LENGKAP KOMENTAR INSTAGRAM")
        print("=" * 70)
        
        try:
            # 1. Load data
            df = self.load_and_clean_data(file_path)
            if df.empty:
                print("❌ Data kosong setelah loading")
                return None, None
            
            # 2. Advanced text cleaning
            df = self.advanced_text_cleaning(df)
            if df.empty:
                print("❌ Data kosong setelah text cleaning")
                return None, None
            
            # 3. Extract text features
            df = self.extract_text_features(df)
            
            # 4. Sentiment analysis
            df = self.perform_sentiment_analysis(df)
            
            # 5. NLP preprocessing
            df = self.nlp_preprocessing(df)
            
            # 6. User behavior analysis
            user_stats = self.user_behavior_analysis(df)
            
            # 7. Create visualizations
            self.create_visualizations(df, user_stats)
            
            # 8. Save processed data
            main_file, user_stats_file = self.save_processed_data(df, user_stats)
            
            print(f"\n🎉 PREPROCESSING SELESAI!")
            print(f"📁 File hasil disimpan di folder 'preprocessing_results/'")
            print(f"📊 Total data processed: {len(df):,} komentar")
            
            return df, user_stats
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None, None

# ============ FUNGSI UTAMA YANG DIKORECT ============

def create_sample_data():
    """Buat data contoh untuk testing"""
    sample_data = {
        'username': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7'],
        'comment': [
            'This is amazing! Love this content! 😍',
            'Not bad, but could be better',
            'What a terrible post, hate this! 😠',
            'Nice photo! Beautiful! 🌸',
            'Okay, nothing special',
            'Awesome work! Keep it up! 💪',
            'Not my favorite, but okay'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_instagram_comments.csv', index=False)
    print("✅ File contoh dibuat: sample_instagram_comments.csv")
    print("📊 Sample data preview:")
    print(df.head())
    return df

def debug_preprocessing():
    """Versi debugging untuk menemukan masalah"""
    print("🐛 DEBUG MODE - Mencari masalah preprocessing")
    
    # Cari file CSV
    import glob
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("❌ Tidak ada file CSV ditemukan di folder ini")
        return None
    
    print("📂 File CSV yang ditemukan:")
    for i, file in enumerate(csv_files, 1):
        print(f"   {i}. {file}")
    
    # Pilih file
    if len(csv_files) == 1:
        file_path = csv_files[0]
    else:
        try:
            choice = int(input("Pilih file (nomor): ")) - 1
            file_path = csv_files[choice]
        except:
            file_path = csv_files[0]
    
    print(f"🔍 Menganalisis file: {file_path}")
    
    try:
        # Coba baca file
        df = pd.read_csv(file_path)
        print(f"✅ Berhasil membaca file: {len(df)} baris, {len(df.columns)} kolom")
        print(f"📋 Kolom: {list(df.columns)}")
        
        # Cek sample data
        print("\n📄 Sample data (3 baris pertama):")
        print(df.head(3).to_string())
        
        # Cek tipe data
        print("\n🔧 Tipe data:")
        print(df.dtypes)
        
        # Cek missing values
        print("\n❓ Missing values:")
        print(df.isnull().sum())
        
        return df
        
    except Exception as e:
        print(f"❌ Error membaca file: {e}")
        return None

def main():
    """Fungsi utama untuk menjalankan preprocessing"""
    preprocessor = InstagramCommentPreprocessor()
    
    # Cari file komentar
    possible_files = [
        "komentar_ig_clean.csv",
        "komentar_ig_raw.csv",
        "instagram_comments.csv", 
        "comments.csv",
        "komentar.csv",
        "sample_instagram_comments.csv",
        "*.csv"  # Cari semua file CSV
    ]
    
    file_found = False
    file_path = None
    
    for file_pattern in possible_files:
        if "*" in file_pattern:
            # Handle wildcard - cari semua file CSV
            import glob
            csv_files = glob.glob("*.csv")
            if csv_files:
                # Prioritaskan file yang namanya mengandung kata kunci komentar
                priority_files = [f for f in csv_files if any(keyword in f.lower() for keyword in ['komentar', 'comment', 'instagram'])]
                if priority_files:
                    file_path = priority_files[0]
                else:
                    file_path = csv_files[0]  # Ambil file CSV pertama
                print(f"📂 Menemukan file: {file_path}")
                file_found = True
                break
        elif os.path.exists(file_pattern):
            print(f"📂 Menemukan file: {file_pattern}")
            file_path = file_pattern
            file_found = True
            break
    
    if not file_found:
        print("❌ Tidak ada file komentar ditemukan")
        print("💡 Membuat data contoh...")
        create_sample_data()
        file_path = "sample_instagram_comments.csv"
        print(f"🔄 Akan menggunakan file contoh: {file_path}")
    
    # Konfirmasi sebelum memproses
    print(f"\n⚠️  Akan memproses file: {file_path}")
    response = input("   Lanjutkan? (y/n): ").lower()
    if response not in ['y', 'yes', 'ya']:
        print("❌ Proses dibatalkan")
        return
    
    # Jalankan preprocessing lengkap
    df_processed, user_stats = preprocessor.run_complete_preprocessing(file_path)
    
    if df_processed is not None:
        print("\n" + "="*60)
        print("💡 PREVIEW DATA HASIL PREPROCESSING:")
        print("="*60)
        
        # Tampilkan preview data
        preview_columns = ['username', 'comment_clean', 'sentiment_vader', 'comment_length']
        available_columns = [col for col in preview_columns if col in df_processed.columns]
        
        if available_columns:
            print(df_processed[available_columns].head(8).to_string(index=False))
        else:
            print("Tidak ada kolom yang tersedia untuk preview")
        
        # Tampilkan statistik singkat
        print(f"\n📊 STATISTIK HASIL PREPROCESSING:")
        print(f"   • Total komentar: {len(df_processed):,}")
        print(f"   • Pengguna unik: {df_processed['username'].nunique():,}")
        
        if 'sentiment_vader' in df_processed.columns:
            sentiment_counts = df_processed['sentiment_vader'].value_counts()
            print(f"   • Sentimen positif: {sentiment_counts.get('positive', 0):,}")
            print(f"   • Sentimen negatif: {sentiment_counts.get('negative', 0):,}")
            print(f"   • Sentimen netral: {sentiment_counts.get('neutral', 0):,}")
        
        print(f"\n✅ Semua proses berhasil! File tersimpan di folder 'preprocessing_results'")
        
    else:
        print("\n❌ Preprocessing gagal. Silakan cek file input dan coba lagi.")

def interactive_main():
    """Mode interaktif untuk user"""
    print("🤖 INSTAGRAM COMMENT PREPROCESSOR")
    print("="*50)
    
    print("Pilih opsi:")
    print("1. Jalankan preprocessing otomatis")
    print("2. Debug mode (untuk troubleshooting)")
    print("3. Buat data contoh")
    print("4. Keluar")
    
    try:
        choice = input("Masukkan pilihan (1/2/3/4): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            debug_preprocessing()
        elif choice == "3":
            create_sample_data()
            print("\n✅ Data contoh telah dibuat. Sekarang jalankan opsi 1 untuk preprocessing.")
        elif choice == "4":
            print("👋 Sampai jumpa!")
            return
        else:
            print("❌ Pilihan tidak valid")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Jalankan mode interaktif
if __name__ == "__main__":
    interactive_main()
