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
                df = pd.read_csv(file_path, encoding='utf-8')
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
            
            # 8. Filter spam comments (opsional)
            spam_words = [
                "follow", "cek bio", "open order", "dm", "giveaway", "promo", 
                "diskon", "langsung wa", "jasa", "jual", "follower", "like back",
                "follow back", "f4f", "l4l", "check my profile", "visit my page"
            ]
            pattern_spam = '|'.join(spam_words)
            initial_count = len(df)
            df = df[~df['comment'].str.contains(pattern_spam, na=False)]
            final_count = len(df)
            print(f"✅ Komentar spam dihapus: {initial_count - final_count}")
            
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
            df['sentence_count'] = df['comment'].apply(
                lambda x: len(re.findall(r'[.!?]+', str(x)))
            )
            
            # 3. Punctuation and capitalization features
            df['exclamation_count'] = df['comment'].str.count('!')
            df['question_count'] = df['comment'].str.count(r'\?')
            df['capital_ratio'] = df['comment'].apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
            )
            
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
            
            # Lemmatization dengan error handling
            if self.lemmatizer is not None:
                try:
                    df['tokens_lemmatized'] = df['tokens_clean'].apply(
                        lambda tokens: [self.lemmatizer.lemmatize(token) for token in tokens]
                    )
                except Exception as e:
                    print(f"⚠️  Lemmatization failed, using cleaned tokens: {e}")
                    df['tokens_lemmatized'] = df['tokens_clean']
            else:
                df['tokens_lemmatized'] = df['tokens_clean']
            
            # Reconstruct cleaned text untuk analisis
            df['comment_clean'] = df['tokens_lemmatized'].apply(
                lambda tokens: ' '.join(tokens) if tokens else ''
            )
            
            # Additional NLP features
            df['unique_word_ratio'] = df['tokens_clean'].apply(
                lambda tokens: len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
            )
            
            print("✅ NLP preprocessing completed")
            return df
            
        except Exception as e:
            print(f"❌ Error dalam NLP preprocessing: {e}")
            # Fallback minimal
            df['tokens'] = df['comment'].apply(self.simple_tokenize)
            df['tokens_clean'] = df['tokens']
            df['tokens_lemmatized'] = df['tokens']
            df['comment_clean'] = df['comment']
            df['unique_word_ratio'] = 0.0
            return df
    
    def user_behavior_analysis(self, df):
        """Analisis perilaku pengguna dengan error handling"""
        print("\n👤 ANALYZING USER BEHAVIOR...")
        
        try:
            # User statistics
            user_stats = df.groupby('username').agg({
                'comment': 'count',
                'comment_length': ['mean', 'std'],
                'vader_compound': 'mean',
                'has_emoji_original': 'mean',
                'has_hashtag_original': 'mean',
                'has_mention_original': 'mean'
            }).round(3)
            
            # Flatten column names
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
            user_stats = user_stats.rename(columns={
                'comment_count': 'comment_count',
                'comment_length_mean': 'avg_comment_length',
                'comment_length_std': 'std_comment_length',
                'vader_compound_mean': 'avg_sentiment',
                'has_emoji_original_mean': 'emoji_ratio',
                'has_hashtag_original_mean': 'hashtag_ratio',
                'has_mention_original_mean': 'mention_ratio'
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
            return pd.DataFrame(columns=['comment_count', 'avg_comment_length', 'std_comment_length', 
                                       'avg_sentiment', 'emoji_ratio', 'hashtag_ratio', 
                                       'mention_ratio', 'user_type'])
    
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
            
            # 4. Social media features
            try:
                plt.subplot(2, 3, 4)
                features = ['has_emoji_original', 'has_hashtag_original', 'has_mention_original']
                feature_counts = [df[feature].sum() for feature in features]
                plt.bar(features, feature_counts, color=['orange', 'purple', 'brown'])
                plt.title('Social Media Features')
                plt.xticks(rotation=45)
                plt.ylabel('Count')
            except Exception as e:
                print(f"⚠️  Error creating social media features chart: {e}")
                plt.subplot(2, 3, 4)
                plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                plt.title('Social Media Features')
            
            # 5. User type distribution
            try:
                plt.subplot(2, 3, 5)
                if 'user_type' in user_stats.columns and not user_stats.empty:
                    user_type_counts = user_stats['user_type'].value_counts()
                    plt.bar(user_type_counts.index, user_type_counts.values, 
                           color=['lightblue', 'orange', 'green'])
                    plt.title('User Type Distribution')
                    plt.ylabel('Number of Users')
                else:
                    plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                    plt.title('User Type Distribution')
            except Exception as e:
                print(f"⚠️  Error creating user type chart: {e}")
                plt.subplot(2, 3, 5)
                plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                plt.title('User Type Distribution')
            
            # 6. Simple sentiment distribution
            try:
                plt.subplot(2, 3, 6)
                sentiment_simple_counts = df['sentiment_simple'].value_counts()
                plt.pie(sentiment_simple_counts.values, labels=sentiment_simple_counts.index,
                        autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
                plt.title('Sentiment Distribution (Simple)')
            except Exception as e:
                print(f"⚠️  Error creating simple sentiment pie chart: {e}")
                plt.subplot(2, 3, 6)
                plt.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center')
                plt.title('Sentiment Distribution (Simple)')
            
            plt.tight_layout()
            plt.savefig('preprocessing_results/basic_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Basic visualizations saved")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def word_frequency_analysis(self, df, top_n=20):
        """Analisis frekuensi kata dengan error handling"""
        print(f"\n📝 ANALYZING WORD FREQUENCY (Top {top_n})...")
        
        try:
            # Gabungkan semua tokens
            all_tokens = []
            for tokens in df.get('tokens_lemmatized', df.get('tokens_clean', [])):
                if isinstance(tokens, list):
                    all_tokens.extend(tokens)
            
            if not all_tokens:
                print("⚠️  No tokens available for frequency analysis")
                return {}, pd.DataFrame()
            
            # Hitung frekuensi manual
            from collections import Counter
            freq_dist = Counter(all_tokens)
            
            # Plot top words
            plt.figure(figsize=(12, 6))
            top_words = freq_dist.most_common(top_n)
            
            words, frequencies = zip(*top_words)
            plt.barh(range(len(words)), frequencies, color='skyblue')
            plt.xlabel('Frequency')
            plt.title(f'Top {top_n} Most Frequent Words')
            plt.yticks(range(len(words)), words)
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('preprocessing_results/word_frequency.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Tampilkan dalam tabel
            top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            print("📊 TOP WORDS:")
            print(top_words_df.to_string(index=False))
            
            return freq_dist, top_words_df
            
        except Exception as e:
            print(f"❌ Error in word frequency analysis: {e}")
            return {}, pd.DataFrame()
    
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
                'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
                'has_emoji_original', 'has_hashtag_original', 'has_mention_original'
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
    
    def generate_summary_report(self, df, user_stats, freq_dist):
        """Buat laporan summary dengan error handling"""
        print("\n📋 GENERATING SUMMARY REPORT...")
        
        try:
            # Prepare data for report
            total_comments = len(df)
            unique_users = df['username'].nunique()
            
            # Sentiment counts dengan error handling
            try:
                vader_counts = df['sentiment_vader'].value_counts()
                vader_text = vader_counts.to_string()
            except:
                vader_text = "Data tidak tersedia"
            
            try:
                simple_counts = df['sentiment_simple'].value_counts() 
                simple_text = simple_counts.to_string()
            except:
                simple_text = "Data tidak tersedia"
            
            # Average scores dengan error handling
            try:
                avg_compound = df['vader_compound'].mean()
                avg_positive = df['vader_positive'].mean()
                avg_negative = df['vader_negative'].mean()
                avg_neutral = df['vader_neutral'].mean()
            except:
                avg_compound = avg_positive = avg_negative = avg_neutral = 0.0
            
            # User stats dengan error handling
            try:
                if not user_stats.empty and 'user_type' in user_stats.columns:
                    user_type_dist = user_stats['user_type'].value_counts().to_string()
                else:
                    user_type_dist = "Data tidak tersedia"
            except:
                user_type_dist = "Data tidak tersedia"
            
            # Social media features
            try:
                emoji_count = df['has_emoji_original'].sum()
                emoji_percent = (emoji_count / total_comments) * 100
                hashtag_count = df['has_hashtag_original'].sum()
                hashtag_percent = (hashtag_count / total_comments) * 100
                mention_count = df['has_mention_original'].sum()
                mention_percent = (mention_count / total_comments) * 100
            except:
                emoji_count = hashtag_count = mention_count = 0
                emoji_percent = hashtag_percent = mention_percent = 0.0
            
            # Top words
            try:
                top_words_text = self.get_top_words_text(freq_dist, 10)
            except:
                top_words_text = "Data tidak tersedia"
            
            report_content = f"""
INSTAGRAM COMMENT PREPROCESSING REPORT
========================================

TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA OVERVIEW:
--------------
• Total Comments Processed: {total_comments:,}
• Unique Users: {unique_users:,}
• Average Comment Length: {df['comment_length'].mean():.1f} characters
• Average Word Count: {df['word_count'].mean():.1f} words

SENTIMENT ANALYSIS:
-------------------
VADER Sentiment Distribution:
{vader_text}

Simple Sentiment Distribution:
{simple_text}

Average VADER Scores:
• Compound: {avg_compound:.3f}
• Positive: {avg_positive:.3f}
• Negative: {avg_negative:.3f}
• Neutral: {avg_neutral:.3f}

USER BEHAVIOR:
--------------
• Total Users: {len(user_stats) if not user_stats.empty else 0:,}
• User Type Distribution:
{user_type_dist}

SOCIAL MEDIA FEATURES:
----------------------
• Comments with Emoji: {emoji_count:,} ({emoji_percent:.1f}%)
• Comments with Hashtag: {hashtag_count:,} ({hashtag_percent:.1f}%)
• Comments with Mention: {mention_count:,} ({mention_percent:.1f}%)

TOP 10 WORDS:
-------------
{top_words_text}

RECOMMENDATIONS:
----------------
1. Content Strategy: Analyze sentiment patterns for engagement optimization
2. User Engagement: Focus on active user retention
3. Content Timing: Post when engagement is highest

---
Generated by Instagram Comment Preprocessing System
"""
            
            report_filename = f"preprocessing_results/preprocessing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"✅ Summary report saved: {report_filename}")
            return report_filename
            
        except Exception as e:
            print(f"❌ Error generating summary report: {e}")
            return None
    
    def get_top_words_text(self, freq_dist, top_n=10):
        """Dapatkan text untuk top words dengan error handling"""
        try:
            if not freq_dist:
                return "No frequency data available"
            
            top_words = freq_dist.most_common(top_n)
            text_lines = []
            for i, (word, freq) in enumerate(top_words, 1):
                text_lines.append(f"{i:2d}. {word:15} : {freq:4} occurrences")
            return '\n'.join(text_lines)
        except:
            return "Error generating top words"
    
    def run_complete_preprocessing(self, file_path):
        """Jalankan preprocessing lengkap dengan comprehensive error handling"""
        print("🚀 MEMULAI PREPROCESSING LENGKAP KOMENTAR INSTAGRAM")
        print("=" * 70)
        
        try:
            # 1. Load data
            df = self.load_and_clean_data(file_path)
            if df.empty:
                print("❌ Data kosong setelah loading")
                return None, None, None
            
            # 2. Advanced text cleaning
            df = self.advanced_text_cleaning(df)
            if df.empty:
                print("❌ Data kosong setelah text cleaning")
                return None, None, None
            
            # 3. Extract text features
            df = self.extract_text_features(df)
            
            # 4. Sentiment analysis
            df = self.perform_sentiment_analysis(df)
            
            # 5. NLP preprocessing
            df = self.nlp_preprocessing(df)
            
            # 6. User behavior analysis
            user_stats = self.user_behavior_analysis(df)
            
            # 7. Word frequency analysis
            freq_dist, top_words = self.word_frequency_analysis(df)
            
            # 8. Create visualizations
            self.create_visualizations(df, user_stats)
            
            # 9. Save processed data
            main_file, user_stats_file = self.save_processed_data(df, user_stats)
            
            # 10. Generate summary report
            report_file = self.generate_summary_report(df, user_stats, freq_dist)
            
            print(f"\n🎉 PREPROCESSING SELESAI!")
            print(f"📁 File hasil disimpan di folder 'preprocessing_results/'")
            print(f"📊 Total data processed: {len(df):,} komentar")
            
            return df, user_stats, freq_dist
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

# ============ CONTOH PENGGUNAAN ============

def main():
    """Contoh penggunaan preprocessor dengan improved file detection"""
    
    # Inisialisasi preprocessor
    preprocessor = InstagramCommentPreprocessor()
    
    # Cari file komentar dengan berbagai pattern
    possible_files = [
        "komentar_ig_clean.csv",
        "komentar_ig_raw.csv",
        "instagram_comments.csv", 
        "comments.csv",
        "komentar.csv",
        "data"
    ]