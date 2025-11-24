import pandas as pd
import time
import random
import re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import os

# ============ KONFIGURASI ============
user_ig = ""
pass_ig = ""
post_url = "https://www.instagram.com/p/DRPEqK6DzQP/"
TARGET_COMMENTS = 200

def setup_driver():
    """Setup Chrome driver dengan konfigurasi yang lebih baik"""
    service = Service(r"C:\Users\User\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe")
    
    options = Options()
    
    # Options untuk menghindari deteksi
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Performance options
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    
    # User agent dan window options
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    options.add_argument('--start-maximized')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-notifications')
    
    # Untuk development, bisa di-comment jika ingin headless
    # options.add_argument('--headless')
    
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def human_like_delay(min_sec=1, max_sec=3):
    """Delay seperti manusia"""
    time.sleep(random.uniform(min_sec, max_sec))

def handle_popups(driver):
    """Handle berbagai popup setelah login"""
    popup_selectors = [
        # Not now untuk save login info
        "//button[contains(text(), 'Not Now')]",
        "//button[contains(text(), 'Lain Kali')]",
        "//button[contains(text(), 'Cancel')]",
        "//button[contains(text(), 'Batal')]",
        # Notifikasi popup
        "//button[contains(text(), 'Not Now') and contains(@class, 'HoLwm')]",
        "//button[contains(., 'Close')]",
        "//button[contains(@aria-label, 'Close')]",
        "//div[contains(@role, 'button') and contains(., 'Not Now')]"
    ]
    
    popups_closed = 0
    for selector in popup_selectors:
        try:
            btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
            driver.execute_script("arguments[0].click();", btn)
            print(f"✅ Popup ditutup: {selector[:50]}...")
            popups_closed += 1
            human_like_delay(1, 2)
        except:
            continue
    
    if popups_closed > 0:
        print(f"✅ Total {popups_closed} popup ditutup")

def login_instagram_improved(driver, username, password):
    """Login ke Instagram dengan error handling yang lebih baik"""
    print("🔄 Melakukan login...")
    
    try:
        driver.get("https://www.instagram.com/accounts/login/")
        human_like_delay(3, 5)
        
        # Tunggu halaman login load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Cari username field dengan multiple selectors
        username_selectors = [
            (By.NAME, "username"),
            (By.XPATH, "//input[@name='username']"),
            (By.XPATH, "//input[@aria-label='Phone number, username, or email']"),
            (By.XPATH, "//input[@placeholder='Phone number, username, or email']")
        ]
        
        username_field = None
        for selector in username_selectors:
            try:
                username_field = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(selector)
                )
                break
            except:
                continue
        
        if not username_field:
            raise Exception("Username field tidak ditemukan")
            
        username_field.clear()
        for char in username:
            username_field.send_keys(char)
            time.sleep(random.uniform(0.1, 0.3))
        human_like_delay(1, 2)
        
        # Password field
        password_selectors = [
            (By.NAME, "password"),
            (By.XPATH, "//input[@name='password']"),
            (By.XPATH, "//input[@aria-label='Password']"),
            (By.XPATH, "//input[@placeholder='Password']")
        ]
        
        password_field = None
        for selector in password_selectors:
            try:
                password_field = driver.find_element(*selector)
                break
            except:
                continue
        
        if not password_field:
            raise Exception("Password field tidak ditemukan")
            
        password_field.clear()
        for char in password:
            password_field.send_keys(char)
            time.sleep(random.uniform(0.1, 0.3))
        human_like_delay(1, 2)
        
        # Login button
        login_selectors = [
            (By.XPATH, "//button[@type='submit']"),
            (By.XPATH, "//div[contains(text(), 'Log In')]"),
            (By.XPATH, "//button[contains(., 'Log In')]"),
            (By.XPATH, "//button[contains(., 'Log in')]")
        ]
        
        login_btn = None
        for selector in login_selectors:
            try:
                login_btn = driver.find_element(*selector)
                if login_btn.is_displayed() and login_btn.is_enabled():
                    break
            except:
                continue
        
        if login_btn:
            driver.execute_script("arguments[0].click();", login_btn)
            print("✅ Login button diklik")
        else:
            raise Exception("Login button tidak ditemukan")
            
        human_like_delay(8, 12)
        
        # Handle popups setelah login
        handle_popups(driver)
        
        # Verifikasi login berhasil
        if "accounts/login" not in driver.current_url:
            print("✅ Login berhasil!")
            return True
        else:
            print("❌ Login mungkin gagal, masih di halaman login")
            return False
        
    except Exception as e:
        print(f"❌ Error login: {e}")
        # Coba screenshot untuk debug
        try:
            driver.save_screenshot("login_error.png")
            print("📸 Screenshot disimpan sebagai 'login_error.png'")
        except:
            pass
        return False

def open_comments_section(driver):
    """Buka section komentar dengan approach yang lebih reliable"""
    print("🔄 Membuka komentar...")
    
    # Strategy 1: Coba klik tombol komentar langsung
    comment_buttons = [
        "//span[contains(@class, 'x1lliihq') and contains(text(), 'comment')]",
        "//button[contains(@class, 'x1n2onr6')]//span[contains(text(), 'comment')]",
        "//a[contains(@href, 'comments')]",
        "//span[contains(., 'komentar')]",
        "//button[.//span[contains(text(), 'comment')]]",
        "//article//button[position()=3]",  # Biasanya tombol ketiga adalah komentar
        "//section//button[.//*[local-name()='svg']]",  # Button dengan SVG icon
    ]
    
    for selector in comment_buttons:
        try:
            element = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
            driver.execute_script("arguments[0].click();", element)
            print(f"✅ Klik komentar berhasil: {selector[:50]}...")
            human_like_delay(3, 5)
            return True
        except:
            continue
    
    # Strategy 2: Coba buka melalui URL langsung
    try:
        if '/p/' in driver.current_url:
            post_id = driver.current_url.split('/p/')[1].split('/')[0]
            comments_url = f"https://www.instagram.com/p/{post_id}/comments/"
            driver.get(comments_url)
            human_like_delay(5, 8)
            print("✅ Buka komentar via URL langsung")
            return True
    except Exception as e:
        print(f"⚠️  Gagal buka via URL: {e}")
    
    print("⚠️  Gagal buka modal komentar, lanjut tanpa modal")
    return False

def scroll_comments_section(driver):
    """Scroll section komentar"""
    try:
        # Coba scroll di modal komentar
        modal_selectors = [
            "//div[contains(@role, 'dialog')]",
            "//section[contains(@class, 'comments')]",
            "//div[contains(@class, '_aano')]",
            "//div[contains(@style, 'overflow')]"
        ]
        
        for selector in modal_selectors:
            try:
                scroll_area = driver.find_element(By.XPATH, selector)
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_area)
                print("   🔄 Scroll di area komentar")
                return True
            except:
                continue
        
        # Fallback: scroll normal
        driver.execute_script("window.scrollBy(0, 500);")
        print("   🔄 Scroll normal")
        return True
        
    except Exception as e:
        print(f"   ⚠️  Error scroll: {e}")
        return False

def is_ui_element(text):
    """Cek apakah text adalah UI element"""
    if not text or len(text.strip()) == 0:
        return True
        
    ui_indicators = [
        'like', 'comment', 'share', 'reply', 'view', 'load more', 'view more',
        'add comment', 'write comment', 'komentar', 'lihat', 'balas', 'send',
        'likes', 'comments', 'replies', 'following', 'follow', 'instagram',
        'post', 'story', 'reel', 'loading', '...', 'caption', 'see translation',
        'posted', 'days ago', 'hours ago', 'minutes ago', 'weeks ago'
    ]
    
    text_lower = text.lower().strip()
    
    # Cek panjang text
    if len(text_lower) < 2 or len(text_lower) > 500:
        return True
    
    # Cek UI indicators
    if any(indicator in text_lower for indicator in ui_indicators):
        return True
    
    # Cek jika hanya berisi simbol atau angka saja
    if re.match(r'^[^\w\s]+$', text_lower) or re.match(r'^\d+$', text_lower):
        return True
    
    return False

def is_valid_username(username):
    """Validasi username"""
    if not username or len(username) < 2 or len(username) > 30:
        return False
    
    # Pattern untuk username Instagram
    if not re.match(r'^[a-zA-Z0-9._]+$', username):
        return False
    
    # Exclude common UI texts
    excluded = ['instagram', 'like', 'comment', 'share', 'view', 'reply', 
                'load', 'more', 'post', 'story', 'reel', 'verified']
    
    if username.lower() in excluded:
        return False
    
    return True

def find_username_nearby(comment_element):
    """Cari username di sekitar element komentar"""
    try:
        # Coba beberapa level parent
        for level in range(1, 6):
            try:
                parent = comment_element.find_element(By.XPATH, f"./ancestor::div[{level}]")
                parent_text = parent.text.strip()
                
                if parent_text:
                    lines = [line.strip() for line in parent_text.split('\n') if line.strip()]
                    
                    for line in lines:
                        if (is_valid_username(line) and 
                            len(line) > 1 and 
                            len(line) < 30 and
                            not is_ui_element(line)):
                            return line
            except:
                continue
                
    except:
        pass
    
    # Fallback: generate username placeholder
    return f"user_{random.randint(1000, 9999)}"

def extract_comments_reliable(driver):
    """Extract komentar dengan method yang lebih reliable"""
    print("📖 Mengekstrak komentar...")
    
    comments_data = []
    max_attempts = 15
    last_count = 0
    no_progress_count = 0
    
    for attempt in range(max_attempts):
        current_count = len(comments_data)
        print(f"   🔍 Attempt {attempt + 1}/{max_attempts}, komentar: {current_count}")
        
        try:
            # Multiple selectors untuk komentar
            comment_selectors = [
                "//div[contains(@class, '_a9zr')]",
                "//article//span[@dir='auto']",
                "//div[contains(@role, 'dialog')]//span[@dir='auto']",
                "//section[contains(@class, 'comments')]//span[@dir='auto']",
                "//div[contains(@class, 'x1lliihq') and string-length(text()) > 5]",
                "//span[@dir='auto' and string-length(text()) > 3]",
            ]
            
            for selector in comment_selectors:
                try:
                    comment_elements = driver.find_elements(By.XPATH, selector)
                    print(f"      Found {len(comment_elements)} elements with {selector[:30]}...")
                    
                    for element in comment_elements:
                        try:
                            comment_text = element.text.strip()
                            if (comment_text and 
                                len(comment_text) > 3 and 
                                not is_ui_element(comment_text)):
                                
                                # Cari username dari parent atau sibling
                                username = find_username_nearby(element)
                                
                                if username and is_valid_username(username):
                                    comment_data = [username, comment_text]
                                    key = (username.lower(), comment_text.lower())
                                    
                                    # Cek duplikat
                                    is_duplicate = any(
                                        existing[0].lower() == username.lower() and 
                                        existing[1].lower() == comment_text.lower() 
                                        for existing in comments_data
                                    )
                                    
                                    if not is_duplicate:
                                        comments_data.append(comment_data)
                                        print(f"      ✅ Komentar {len(comments_data)}: @{username}")
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    continue
            
            # Progress check
            if len(comments_data) > last_count:
                last_count = len(comments_data)
                no_progress_count = 0
            else:
                no_progress_count += 1
            
            # Jika sudah dapat cukup komentar, break
            if len(comments_data) >= TARGET_COMMENTS:
                print(f"🎯 Target {TARGET_COMMENTS} komentar tercapai!")
                break
                
            # Jika stuck, coba scroll
            if no_progress_count >= 3:
                print("   🔄 Stuck, scrolling...")
                scroll_comments_section(driver)
                human_like_delay(2, 3)
                no_progress_count = 0
            else:
                # Scroll untuk load lebih banyak
                scroll_comments_section(driver)
                human_like_delay(1, 2)
                
        except Exception as e:
            print(f"   ⚠️  Error ekstraksi attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                break
            human_like_delay(1, 2)
    
    return comments_data

def save_comments_to_file(df, filename=None):
    """Save DataFrame komentar ke file"""
    if df.empty:
        print("❌ Tidak ada data untuk disimpan")
        return None
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"instagram_comments_{timestamp}.csv"
    
    try:
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"💾 Data disimpan: {filename}")
        print(f"📊 Total komentar: {len(df)}")
        return filename
    except Exception as e:
        print(f"❌ Error menyimpan file: {e}")
        return None

def clean_comments_data(comments_data):
    """Bersihkan data komentar dari duplikat dan data tidak valid"""
    print("🧹 Membersihkan data komentar...")
    
    unique_comments = []
    seen = set()
    
    for comment in comments_data:
        if len(comment) != 2:
            continue
            
        username, comment_text = comment
        
        # Skip jika username atau comment kosong
        if not username or not comment_text:
            continue
            
        # Skip jika username sama dengan comment (kemungkinan data tidak valid)
        if username.strip() == comment_text.strip():
            continue
            
        # Skip komentar yang terlalu pendek
        if len(comment_text.strip()) < 3:
            continue
            
        key = (username.lower().strip(), comment_text.lower().strip())
        
        if key not in seen:
            seen.add(key)
            unique_comments.append([username.strip(), comment_text.strip()])
    
    print(f"✅ Data bersih: {len(unique_comments)} komentar (setelah cleaning)")
    return unique_comments

def main():
    print("🚀 MEMULAI SCRAPING KOMENTAR INSTAGRAM")
    print("=" * 50)
    print(f"🎯 Target: {TARGET_COMMENTS} komentar")
    print("=" * 50)
    
    driver = None
    try:
        # Setup driver
        driver = setup_driver()
        
        # Login
        if not login_instagram_improved(driver, user_ig, pass_ig):
            print("❌ Login gagal, keluar...")
            return pd.DataFrame()
        
        # Navigate ke post
        print(f"🌐 Mengakses post: {post_url}")
        driver.get(post_url)
        human_like_delay(5, 8)
        
        # Verifikasi kita di halaman post
        if '/p/' not in driver.current_url:
            print("❌ Gagal mengakses post")
            return pd.DataFrame()
        
        # Buka section komentar
        if not open_comments_section(driver):
            print("⚠️  Lanjut tanpa modal komentar")
        
        # Tunggu sebentar untuk memastikan komentar load
        human_like_delay(3, 5)
        
        # Extract komentar
        comments_data = extract_comments_reliable(driver)
        
        # Clean data
        cleaned_comments = clean_comments_data(comments_data)
        
        # Create DataFrame
        df = pd.DataFrame(cleaned_comments, columns=['username', 'comment'])
        
        # Tampilkan hasil
        print("\n" + "=" * 50)
        print("📊 HASIL SCRAPING")
        print("=" * 50)
        print(f"✅ Berhasil mengumpulkan: {len(df)} komentar")
        
        if not df.empty:
            print("\nPreview komentar:")
            for i, (username, comment) in enumerate(df.head(10).values, 1):
                truncated_comment = comment[:80] + '...' if len(comment) > 80 else comment
                print(f"{i:2d}. @{username}: {truncated_comment}")
            
            # Save to file
            filename = save_comments_to_file(df)
            
            if len(df) < TARGET_COMMENTS:
                print(f"\n⚠️  Hanya mendapatkan {len(df)} dari target {TARGET_COMMENTS} komentar")
                print("💡 Kemungkinan: Post tidak memiliki cukup komentar atau struktur berbeda")
            else:
                print(f"\n🎉 Target {TARGET_COMMENTS} komentar tercapai!")
        else:
            print("❌ Tidak ada komentar yang berhasil di-scrape")
        
        return df
        
    except Exception as e:
        print(f"❌ Error utama: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    finally:
        if driver:
            print("\n🔚 Menutup browser...")
            driver.quit()

if __name__ == "__main__":
    df = main()
    
    if not df.empty:
        print(f"\n🎉 Scraping selesai! {len(df)} komentar berhasil dikumpulkan dan disimpan")
    else:
        print("\n😞 Scraping gagal, tidak ada data yang dikumpulkan")
