# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Data Generation
# MAGIC
# MAGIC **Business Context:** Global Music Rights (GMR) is a performance rights organization that manages
# MAGIC licensing and royalty distribution for over 121K+ songs. This notebook generates a realistic
# MAGIC dataset using **real song metadata from MusicBrainz** combined with synthetic financial data
# MAGIC modeled on published industry rates.
# MAGIC
# MAGIC ## Data Sources
# MAGIC | Source | What We Get | Real or Synthetic? |
# MAGIC |--------|-------------|-------------------|
# MAGIC | MusicBrainz API | Song titles, artists, songwriters, ISRCs, release dates, durations | **Real** |
# MAGIC | Seed list | Genre classification, artist grouping | **Curated** |
# MAGIC | Industry rates | Royalty payments, license fees, performance logs | **Synthetic** |
# MAGIC
# MAGIC ## Tables Created
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `songwriters` | Real songwriter profiles with PRO affiliations |
# MAGIC | `songs` | Real song catalog with MusicBrainz metadata |
# MAGIC | `licenses` | Synthetic licensing agreements |
# MAGIC | `performance_logs` | Synthetic play/performance tracking |
# MAGIC | `royalty_payments` | Synthetic payments based on industry rates |

# COMMAND ----------

# MAGIC %pip install musicbrainzngs --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo_catalog", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

print(f"Creating tables in: {CATALOG}.{SCHEMA}")

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed Song List
# MAGIC
# MAGIC A curated list of well-known songs across genres. MusicBrainz will provide the
# MAGIC real songwriter credits, ISRCs, release dates, and durations.

# COMMAND ----------

# Seed list: just title + artist + genre (metadata comes from MusicBrainz)
SEED_SONGS = [
    # --- POP ---
    ("Blinding Lights", "The Weeknd", "Pop"),
    ("Shape of You", "Ed Sheeran", "Pop"),
    ("Rolling in the Deep", "Adele", "Pop"),
    ("Shake It Off", "Taylor Swift", "Pop"),
    ("Uptown Funk", "Mark Ronson", "Pop"),
    ("Happy", "Pharrell Williams", "Pop"),
    ("Bad Guy", "Billie Eilish", "Pop"),
    ("Levitating", "Dua Lipa", "Pop"),
    ("Watermelon Sugar", "Harry Styles", "Pop"),
    ("Someone Like You", "Adele", "Pop"),
    ("Hello", "Adele", "Pop"),
    ("Poker Face", "Lady Gaga", "Pop"),
    ("Chandelier", "Sia", "Pop"),
    ("drivers license", "Olivia Rodrigo", "Pop"),
    ("Royals", "Lorde", "Pop"),
    ("Stay With Me", "Sam Smith", "Pop"),
    ("Anti-Hero", "Taylor Swift", "Pop"),
    ("Cruel Summer", "Taylor Swift", "Pop"),
    ("As It Was", "Harry Styles", "Pop"),
    ("good 4 u", "Olivia Rodrigo", "Pop"),
    ("Save Your Tears", "The Weeknd", "Pop"),
    ("Just Dance", "Lady Gaga", "Pop"),
    ("24K Magic", "Bruno Mars", "Pop"),
    ("Blank Space", "Taylor Swift", "Pop"),
    ("Starboy", "The Weeknd", "Pop"),
    ("Can't Stop the Feeling!", "Justin Timberlake", "Pop"),
    ("Closer", "The Chainsmokers", "Pop"),

    # --- ROCK ---
    ("Bohemian Rhapsody", "Queen", "Rock"),
    ("Stairway to Heaven", "Led Zeppelin", "Rock"),
    ("Hotel California", "Eagles", "Rock"),
    ("Smells Like Teen Spirit", "Nirvana", "Rock"),
    ("Wonderwall", "Oasis", "Rock"),
    ("Sweet Child O' Mine", "Guns N' Roses", "Rock"),
    ("Yellow", "Coldplay", "Rock"),
    ("Viva la Vida", "Coldplay", "Rock"),
    ("Fix You", "Coldplay", "Rock"),
    ("Creep", "Radiohead", "Rock"),
    ("Under the Bridge", "Red Hot Chili Peppers", "Rock"),
    ("Californication", "Red Hot Chili Peppers", "Rock"),
    ("Black", "Pearl Jam", "Rock"),
    ("With or Without You", "U2", "Rock"),
    ("We Will Rock You", "Queen", "Rock"),
    ("We Are the Champions", "Queen", "Rock"),
    ("Don't Stop Believin'", "Journey", "Rock"),
    ("Everlong", "Foo Fighters", "Rock"),
    ("Come As You Are", "Nirvana", "Rock"),
    ("Enter Sandman", "Metallica", "Rock"),
    ("Back in Black", "AC/DC", "Rock"),
    ("Livin' on a Prayer", "Bon Jovi", "Rock"),

    # --- R&B / HIP-HOP ---
    ("Crazy in Love", "Beyonce", "R&B"),
    ("Lose Yourself", "Eminem", "Hip-Hop"),
    ("Hotline Bling", "Drake", "Hip-Hop"),
    ("HUMBLE.", "Kendrick Lamar", "Hip-Hop"),
    ("Gold Digger", "Kanye West", "Hip-Hop"),
    ("Single Ladies (Put a Ring on It)", "Beyonce", "R&B"),
    ("No One", "Alicia Keys", "R&B"),
    ("If I Ain't Got You", "Alicia Keys", "R&B"),
    ("Halo", "Beyonce", "R&B"),
    ("Stan", "Eminem", "Hip-Hop"),
    ("God's Plan", "Drake", "Hip-Hop"),
    ("Alright", "Kendrick Lamar", "Hip-Hop"),
    ("Kill Bill", "SZA", "R&B"),
    ("Doo Wop (That Thing)", "Lauryn Hill", "R&B"),
    ("Work It", "Missy Elliott", "Hip-Hop"),
    ("Sicko Mode", "Travis Scott", "Hip-Hop"),
    ("Old Town Road", "Lil Nas X", "Hip-Hop"),
    ("Redbone", "Childish Gambino", "R&B"),

    # --- COUNTRY ---
    ("Jolene", "Dolly Parton", "Country"),
    ("Take Me Home, Country Roads", "John Denver", "Country"),
    ("9 to 5", "Dolly Parton", "Country"),
    ("Ring of Fire", "Johnny Cash", "Country"),
    ("On the Road Again", "Willie Nelson", "Country"),
    ("Friends in Low Places", "Garth Brooks", "Country"),
    ("I Walk the Line", "Johnny Cash", "Country"),
    ("Tennessee Whiskey", "Chris Stapleton", "Country"),
    ("Coat of Many Colors", "Dolly Parton", "Country"),
    ("Folsom Prison Blues", "Johnny Cash", "Country"),
    ("The Gambler", "Kenny Rogers", "Country"),
    ("Wagon Wheel", "Darius Rucker", "Country"),
    ("Before He Cheats", "Carrie Underwood", "Country"),

    # --- ELECTRONIC / DANCE ---
    ("Get Lucky", "Daft Punk", "Electronic"),
    ("One More Time", "Daft Punk", "Electronic"),
    ("Levels", "Avicii", "Electronic"),
    ("Wake Me Up", "Avicii", "Electronic"),
    ("Summer", "Calvin Harris", "Electronic"),
    ("Titanium", "David Guetta", "Electronic"),
    ("Lean On", "Major Lazer", "Electronic"),
    ("Happier", "Marshmello", "Electronic"),
    ("Around the World", "Daft Punk", "Electronic"),
    ("Don't You Worry Child", "Swedish House Mafia", "Electronic"),

    # --- JAZZ / LATIN / OTHER ---
    ("Despacito", "Luis Fonsi", "Latin"),
    ("Take Five", "Dave Brubeck", "Jazz"),
    ("Fly Me to the Moon", "Frank Sinatra", "Jazz"),
    ("Three Little Birds", "Bob Marley", "Reggae"),
    ("No Woman, No Cry", "Bob Marley", "Reggae"),
    ("Hips Don't Lie", "Shakira", "Latin"),
    ("Waka Waka", "Shakira", "Latin"),
    ("Livin' la Vida Loca", "Ricky Martin", "Latin"),
    ("What a Wonderful World", "Louis Armstrong", "Jazz"),
    ("My Way", "Frank Sinatra", "Jazz"),
    ("Summertime", "Ella Fitzgerald", "Jazz"),
    ("La Bamba", "Ritchie Valens", "Latin"),
]

print(f"Seed songs: {len(SEED_SONGS)}")

# Genre distribution
from collections import Counter
genre_counts = Counter(g for _, _, g in SEED_SONGS)
for genre, count in genre_counts.most_common():
    print(f"  {genre}: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull Real Metadata from MusicBrainz
# MAGIC
# MAGIC For each seed song, look up the real songwriter credits, ISRC codes,
# MAGIC release dates, and durations from the MusicBrainz open music database.

# COMMAND ----------

import musicbrainzngs
import time

# Required: set a descriptive user agent for MusicBrainz API
musicbrainzngs.set_useragent("GMR-Royalty-Demo", "1.0", "demo@example.com")

def lookup_song(title, artist):
    """
    Look up a song in MusicBrainz and return real metadata.
    Returns: dict with title, artist, writers, isrc, release_date, duration_seconds
    """
    try:
        # Step 1: Search for the recording
        result = musicbrainzngs.search_recordings(
            recording=title, artist=artist, limit=5
        )
        recordings = result.get("recording-list", [])
        if not recordings:
            return None

        # Pick the best match (highest score)
        rec = recordings[0]
        rec_id = rec["id"]
        rec_title = rec.get("title", title)
        rec_artist = rec.get("artist-credit-phrase", artist)
        duration = int(rec["length"]) // 1000 if "length" in rec else None

        time.sleep(1.1)  # Rate limit: max 1 req/sec

        # Step 2: Get recording details (ISRCs, work relations, releases)
        rec_detail = musicbrainzngs.get_recording_by_id(
            rec_id, includes=["isrcs", "work-rels", "releases"]
        )
        recording = rec_detail["recording"]

        # Extract ISRC
        isrcs = recording.get("isrc-list", [])
        isrc = isrcs[0] if isrcs else None

        # Extract release date from first release
        releases = recording.get("release-list", [])
        release_date = None
        for rel in releases:
            if "date" in rel and len(rel["date"]) >= 4:
                release_date = rel["date"]
                break

        # Duration from detailed lookup (more reliable)
        if not duration and "length" in recording:
            duration = int(recording["length"]) // 1000

        # Step 3: Get songwriter credits via work relations
        writers = []
        work_rels = recording.get("work-relation-list", [])
        for wr in work_rels:
            if wr.get("type") == "performance" and "work" in wr:
                work_id = wr["work"]["id"]
                time.sleep(1.1)

                try:
                    work_detail = musicbrainzngs.get_work_by_id(
                        work_id, includes=["artist-rels"]
                    )
                    work = work_detail["work"]
                    for ar in work.get("artist-relation-list", []):
                        rel_type = ar.get("type", "")
                        if rel_type in ("composer", "lyricist", "writer"):
                            writer_name = ar["artist"]["name"]
                            if writer_name not in writers:
                                writers.append(writer_name)
                except Exception:
                    pass
                break  # Only need the first work

        return {
            "title": rec_title,
            "artist": rec_artist,
            "writers": writers,
            "isrc": isrc,
            "release_date": release_date,
            "duration_seconds": duration,
        }

    except Exception as e:
        print(f"  Error looking up '{title}' by {artist}: {e}")
        return None

# COMMAND ----------

# Look up all seed songs
print("Pulling real metadata from MusicBrainz...")
print("(This takes ~3-5 minutes due to API rate limiting)")
print("=" * 60)

enriched_songs = []
failed_songs = []

for i, (title, artist, genre) in enumerate(SEED_SONGS):
    result = lookup_song(title, artist)

    if result and result["writers"]:
        result["genre"] = genre
        result["seed_artist"] = artist  # Keep original for matching
        enriched_songs.append(result)
        writers_str = ", ".join(result["writers"][:3])
        print(f"[{i+1}/{len(SEED_SONGS)}] {title} -> Writers: {writers_str}")
    else:
        # Fallback: use artist as songwriter
        enriched_songs.append({
            "title": title,
            "artist": artist,
            "genre": genre,
            "seed_artist": artist,
            "writers": [artist],  # Use performing artist as fallback
            "isrc": None,
            "release_date": None,
            "duration_seconds": None,
        })
        failed_songs.append(title)
        print(f"[{i+1}/{len(SEED_SONGS)}] {title} -> Fallback (no credits found)")

print(f"\nResults: {len(enriched_songs) - len(failed_songs)} with real credits, {len(failed_songs)} fallback")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Songwriters Table
# MAGIC
# MAGIC Extract all unique songwriters from MusicBrainz results and assign PRO affiliations.

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timedelta
import random

random.seed(42)  # Reproducible

# Extract unique songwriters from all enriched songs
all_writers = {}
for song in enriched_songs:
    for writer_name in song["writers"]:
        if writer_name not in all_writers:
            all_writers[writer_name] = {
                "songs": [],
                "publishers": set(),
            }
        all_writers[writer_name]["songs"].append(song["title"])

# PRO distribution (realistic for US market, with GMR representation for demo)
PRO_AFFILIATIONS = ["GMR", "ASCAP", "BMI", "SESAC"]
PRO_WEIGHTS = [0.20, 0.35, 0.35, 0.10]

# Major publishers
PUBLISHERS = [
    "Universal Music Publishing", "Sony Music Publishing", "Warner Chappell Music",
    "BMG Rights Management", "Kobalt Music", "Concord Music",
    "Spirit Music Group", "Reservoir Media", "Downtown Music Publishing"
]

# Build songwriter records
songwriter_data = []
writer_id_map = {}  # name -> songwriter_id

for i, (name, info) in enumerate(sorted(all_writers.items()), start=1):
    sw_id = f"SW{i:05d}"
    writer_id_map[name] = sw_id

    pro = random.choices(PRO_AFFILIATIONS, weights=PRO_WEIGHTS, k=1)[0]
    ipi = f"{random.randint(100000000, 99999999999):011d}"
    publisher = random.choice(PUBLISHERS)
    split = round(random.uniform(0.25, 0.75), 4)
    # Generate plausible email
    name_parts = name.lower().replace("'", "").replace(".", "").split()
    email = f"{'.'.join(name_parts[:2])}@email.com" if len(name_parts) >= 2 else f"{name_parts[0]}@email.com"

    songwriter_data.append({
        "songwriter_id": sw_id,
        "name": name,
        "pro_affiliation": pro,
        "ipi_number": ipi,
        "split_percentage": split,
        "email": email,
    })

songwriters_df = spark.createDataFrame(songwriter_data)
songwriters_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.songwriters")
print(f"Created songwriters table with {len(songwriter_data)} real songwriters")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.songwriters").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Songs Table
# MAGIC
# MAGIC Create the songs table using real MusicBrainz metadata.

# COMMAND ----------

def generate_isrc():
    """Generate a realistic ISRC code as fallback."""
    country = random.choice(["US", "GB", "CA", "AU"])
    registrant = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
    year = random.randint(18, 25)
    designation = random.randint(10000, 99999)
    return f"{country}{registrant}{year:02d}{designation}"

song_data = []
for i, song in enumerate(enriched_songs, start=1):
    # Map writer names to songwriter IDs
    writer_ids = [writer_id_map[w] for w in song["writers"] if w in writer_id_map]
    if not writer_ids:
        continue

    # Parse release date
    release_date = None
    if song.get("release_date"):
        try:
            date_str = song["release_date"]
            if len(date_str) == 4:  # Just year
                release_date = datetime(int(date_str), 1, 1).date()
            elif len(date_str) == 7:  # Year-month
                release_date = datetime.strptime(date_str, "%Y-%m").date()
            else:
                release_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    if not release_date:
        # Fallback: random date in last 30 years
        days_ago = random.randint(0, 10950)
        release_date = (datetime.now() - timedelta(days=days_ago)).date()

    song_data.append({
        "song_id": f"SONG{i:06d}",
        "title": song["title"],
        "artist_name": song["artist"],
        "songwriters": ",".join(writer_ids),
        "publisher": random.choice(PUBLISHERS),
        "genre": song["genre"],
        "release_date": release_date,
        "isrc_code": song.get("isrc") or generate_isrc(),
        "duration_seconds": song.get("duration_seconds") or random.randint(150, 360),
    })

songs_df = spark.createDataFrame(song_data)
songs_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.songs")
print(f"Created songs table with {len(song_data)} real songs")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.songs").limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Licenses Table
# MAGIC
# MAGIC Synthetic licensing agreements using real licensee names. Fee amounts based
# MAGIC on published industry rates.

# COMMAND ----------

NUM_LICENSES = 600

songs_list = [s["song_id"] for s in song_data]

LICENSEES = [
    "Spotify USA Inc.", "Apple Inc.", "Amazon.com Services LLC", "Pandora Media LLC",
    "iHeartMedia Inc.", "SiriusXM Holdings Inc.", "YouTube LLC", "TikTok Inc.",
    "Entercom Communications", "Cumulus Media Inc.", "Townsquare Media",
    "Live Nation Entertainment", "AEG Presents", "MSG Entertainment",
    "Hilton Worldwide", "Marriott International", "Starbucks Corporation",
    "Target Corporation", "Walmart Inc.", "Peloton Interactive"
]

LICENSE_TYPES = ["streaming", "radio", "live_venue", "digital", "sync", "mechanical"]
TERRITORIES = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "MX", "KR", "IN", "ES", "IT", "NL", "SE"]

# Industry-standard fee ranges by license type
LICENSE_FEE_RANGES = {
    "streaming": (2000, 15000),
    "radio": (1000, 8000),
    "live_venue": (5000, 30000),
    "digital": (1500, 10000),
    "sync": (5000, 500000),
    "mechanical": (500, 5000),
}

license_data = []
for i in range(1, NUM_LICENSES + 1):
    song_id = random.choice(songs_list)
    licensee = random.choice(LICENSEES)
    license_type = random.choice(LICENSE_TYPES)
    territory = random.choice(TERRITORIES)

    start_days_ago = random.randint(0, 1825)
    start_date = datetime.now() - timedelta(days=start_days_ago)
    end_date = start_date + timedelta(days=365 * random.randint(1, 5))

    fee_min, fee_max = LICENSE_FEE_RANGES[license_type]
    fee = round(random.uniform(fee_min, fee_max), 2)

    license_data.append({
        "license_id": f"LIC{i:07d}",
        "song_id": song_id,
        "licensee_name": licensee,
        "license_type": license_type,
        "territory": territory,
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "fee_amount": fee,
    })

licenses_df = spark.createDataFrame(license_data)
licenses_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.licenses")
print(f"Created licenses table with {len(license_data)} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.licenses").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Performance Logs Table
# MAGIC
# MAGIC Synthetic play data distributed evenly across all songs.

# COMMAND ----------

NUM_PERFORMANCE_LOGS = 8000

PLATFORMS = [
    "Spotify", "Apple Music", "Amazon Music", "YouTube Music", "Pandora", "iHeartRadio",
    "SiriusXM", "KROQ-FM", "KIIS-FM", "Z100", "BBC Radio 1", "Capital FM",
    "Madison Square Garden", "Red Rocks Amphitheatre", "The Forum", "Staples Center",
    "House of Blues", "The Roxy", "Bowery Ballroom", "9:30 Club"
]

performance_logs_data = []
for i in range(1, NUM_PERFORMANCE_LOGS + 1):
    days_ago = random.randint(0, 365)
    hours_ago = random.randint(0, 23)
    play_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago)

    performance_logs_data.append({
        "log_id": f"LOG{i:08d}",
        "song_id": random.choice(songs_list),
        "platform": random.choice(PLATFORMS),
        "play_timestamp": play_time,
        "territory": random.choice(TERRITORIES),
        "duration_played": random.randint(30, 420),
        "reported_by": random.choice(LICENSEES[:10]),
    })

performance_logs_df = spark.createDataFrame(performance_logs_data)
performance_logs_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.performance_logs")
print(f"Created performance_logs table with {len(performance_logs_data)} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.performance_logs").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Royalty Payments Table
# MAGIC
# MAGIC Synthetic payments based on published industry rates:
# MAGIC - Streaming: ~$0.003-0.005 per stream (Spotify), ~$0.006-0.01 (Apple Music)
# MAGIC - Radio: ~$0.10-0.50 per performance
# MAGIC - Sync: $5K-$500K per placement
# MAGIC - Live: $50-$5,000 per performance

# COMMAND ----------

NUM_ROYALTY_PAYMENTS = 2000

payment_periods = []
for year in range(2022, 2026):
    for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        payment_periods.append(f"{year}-{quarter}")

PAYMENT_STATUSES = ["completed", "pending", "processing", "disputed", "on_hold"]

# Map song_id -> writer IDs
song_to_writers = {s["song_id"]: s["songwriters"].split(",") for s in song_data}

royalty_data = []
for i in range(1, NUM_ROYALTY_PAYMENTS + 1):
    song_id = random.choice(songs_list)
    songwriter_id = random.choice(song_to_writers[song_id])
    period = random.choice(payment_periods)

    # Realistic quarterly royalty range ($100 - $15,000 per writer per quarter)
    gross_amount = round(random.uniform(100, 15000), 2)
    deduction_rate = random.uniform(0.10, 0.25)
    deductions = round(gross_amount * deduction_rate, 2)
    net_amount = round(gross_amount - deductions, 2)

    status = random.choices(
        PAYMENT_STATUSES,
        weights=[0.80, 0.08, 0.06, 0.04, 0.02]
    )[0]

    year, quarter = period.split("-")
    quarter_end_month = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}[quarter]
    payment_date = datetime(int(year), quarter_end_month, 28) + timedelta(days=random.randint(30, 90))

    royalty_data.append({
        "payment_id": f"PAY{i:07d}",
        "song_id": song_id,
        "songwriter_id": songwriter_id,
        "payment_period": period,
        "gross_amount": gross_amount,
        "deductions": deductions,
        "net_amount": net_amount,
        "payment_status": status,
        "payment_date": payment_date.date() if status == "completed" else None,
    })

royalty_payments_df = spark.createDataFrame(royalty_data)
royalty_payments_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.royalty_payments")
print(f"Created royalty_payments table with {len(royalty_data)} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.royalty_payments").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Summary

# COMMAND ----------

tables = ["songwriters", "songs", "licenses", "performance_logs", "royalty_payments"]
summary_data = []

for table in tables:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table}").count()
    summary_data.append({"table": table, "row_count": count})

summary_df = spark.createDataFrame(summary_data)
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC The data generation is complete. Proceed to:
# MAGIC 1. **02_data_ingestion.py** - Demonstrate Auto Loader patterns
# MAGIC 2. **03_feature_engineering.py** - Create Feature Store tables
# MAGIC 3. **04_vector_index.py** - Build vector search index for semantic song discovery

# COMMAND ----------

print(f"""
GMR Demo Data Generation Complete!
==================================
Catalog: {CATALOG}
Schema: {SCHEMA}

Data sources:
- Song metadata: MusicBrainz (real)
- Financial data: Synthetic (industry rates)

Tables created:
- {CATALOG}.{SCHEMA}.songwriters ({len(songwriter_data)} real songwriters)
- {CATALOG}.{SCHEMA}.songs ({len(song_data)} real songs)
- {CATALOG}.{SCHEMA}.licenses ({NUM_LICENSES} synthetic)
- {CATALOG}.{SCHEMA}.performance_logs ({NUM_PERFORMANCE_LOGS} synthetic)
- {CATALOG}.{SCHEMA}.royalty_payments ({NUM_ROYALTY_PAYMENTS} synthetic)
""")
