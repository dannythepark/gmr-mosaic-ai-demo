# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Data Generation
# MAGIC
# MAGIC **Business Context:** Global Music Rights (GMR) is a performance rights organization that manages licensing
# MAGIC and royalty distribution for over 121K+ songs. This notebook generates realistic sample data that mirrors
# MAGIC GMR's domain to demonstrate the full Mosaic AI agent development lifecycle.
# MAGIC
# MAGIC ## Tables Created
# MAGIC | Table | Description | Row Count |
# MAGIC |-------|-------------|-----------|
# MAGIC | `songwriters` | Songwriter profiles with PRO affiliations | 200+ |
# MAGIC | `songs` | Song catalog with metadata | 500+ |
# MAGIC | `licenses` | Licensing agreements by territory and type | 2,000+ |
# MAGIC | `performance_logs` | Play/performance tracking data | 50,000+ |
# MAGIC | `royalty_payments` | Payment records to songwriters | 5,000+ |

# COMMAND ----------

# MAGIC %pip install dbldatagen --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
import random

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Widget parameters - can be overridden at runtime
dbutils.widgets.text("catalog", "gmr_demo", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

print(f"Creating tables in: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Catalog and Schema

# COMMAND ----------

# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference Data for Realistic Generation

# COMMAND ----------

# GMR-relevant reference data
GENRES = ["Pop", "Rock", "Hip-Hop", "R&B", "Country", "Electronic", "Jazz", "Classical", "Latin", "Indie"]
PRO_AFFILIATIONS = ["GMR", "ASCAP", "BMI", "SESAC"]  # Performance Rights Organizations
LICENSE_TYPES = ["radio", "streaming", "live_venue", "digital", "sync", "mechanical"]
TERRITORIES = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "MX", "KR", "IN", "ES", "IT", "NL", "SE"]
PLATFORMS = [
    "Spotify", "Apple Music", "Amazon Music", "YouTube Music", "Pandora", "iHeartRadio",
    "SiriusXM", "KROQ-FM", "KIIS-FM", "Z100", "BBC Radio 1", "Capital FM",
    "Madison Square Garden", "Red Rocks Amphitheatre", "The Forum", "Staples Center",
    "House of Blues", "The Roxy", "Bowery Ballroom", "9:30 Club"
]
PUBLISHERS = [
    "Universal Music Publishing", "Sony Music Publishing", "Warner Chappell Music",
    "BMG Rights Management", "Kobalt Music", "Concord Music", "Downtown Music",
    "Spirit Music Group", "Reservoir Media", "Round Hill Music"
]

# First and last names for realistic songwriter generation
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Marcus", "Aaliyah", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker"
]

# Song title components for realistic title generation
TITLE_ADJECTIVES = ["Midnight", "Golden", "Electric", "Broken", "Endless", "Wild", "Silent", "Burning", "Fading", "Rising"]
TITLE_NOUNS = ["Dreams", "Hearts", "Lights", "Shadows", "Fire", "Rain", "Stars", "Love", "Time", "Night"]
TITLE_TEMPLATES = [
    "{adj} {noun}", "The {noun}", "{noun} of {noun}", "{adj} {noun}s",
    "When {noun} Falls", "Into the {noun}", "{adj}", "Last {noun}",
    "Dancing in the {noun}", "{noun} Never Dies"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Songwriters Table
# MAGIC
# MAGIC As a royalty analyst at GMR, understanding songwriter profiles and their PRO affiliations
# MAGIC is critical for accurate payment distribution.

# COMMAND ----------

NUM_SONGWRITERS = 200

# Generate songwriter data
songwriter_data = []
for i in range(1, NUM_SONGWRITERS + 1):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    pro = random.choice(PRO_AFFILIATIONS)
    # IPI numbers are 9-11 digits
    ipi = f"{random.randint(100000000, 99999999999):011d}"
    # Split percentage typically ranges from 5% to 100% (for sole writers)
    split = round(random.uniform(0.05, 1.0), 4)
    email = f"{first.lower()}.{last.lower()}@email.com"

    songwriter_data.append({
        "songwriter_id": f"SW{i:05d}",
        "name": name,
        "pro_affiliation": pro,
        "ipi_number": ipi,
        "split_percentage": split,
        "email": email
    })

songwriters_df = spark.createDataFrame(songwriter_data)

# Write to Delta table
songwriters_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.songwriters")
print(f"Created songwriters table with {songwriters_df.count()} records")

# COMMAND ----------

# Display sample
display(spark.table(f"{CATALOG}.{SCHEMA}.songwriters").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Songs Table
# MAGIC
# MAGIC The song catalog is the core asset that GMR manages - each song has associated
# MAGIC ISRC codes, songwriter splits, and publisher relationships.

# COMMAND ----------

NUM_SONGS = 500

# Load songwriters for referencing
songwriters_list = [row.songwriter_id for row in spark.table(f"{CATALOG}.{SCHEMA}.songwriters").collect()]

def generate_title():
    template = random.choice(TITLE_TEMPLATES)
    return template.format(
        adj=random.choice(TITLE_ADJECTIVES),
        noun=random.choice(TITLE_NOUNS)
    )

def generate_isrc():
    # ISRC format: CC-XXX-YY-NNNNN (Country-Registrant-Year-Designation)
    country = random.choice(["US", "GB", "CA", "AU"])
    registrant = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
    year = random.randint(18, 25)
    designation = random.randint(10000, 99999)
    return f"{country}{registrant}{year:02d}{designation}"

# Generate song data
song_data = []
for i in range(1, NUM_SONGS + 1):
    title = generate_title()
    # Assign 1-3 songwriters per song
    num_writers = random.randint(1, 3)
    writers = random.sample(songwriters_list, num_writers)

    # Generate release date within last 10 years
    days_ago = random.randint(0, 3650)
    release_date = datetime.now() - timedelta(days=days_ago)

    song_data.append({
        "song_id": f"SONG{i:06d}",
        "title": title,
        "artist_name": f"Artist {random.randint(1, 100)}",
        "songwriters": ",".join(writers),
        "publisher": random.choice(PUBLISHERS),
        "genre": random.choice(GENRES),
        "release_date": release_date.date(),
        "isrc_code": generate_isrc(),
        "duration_seconds": random.randint(120, 420)  # 2-7 minutes
    })

songs_df = spark.createDataFrame(song_data)

# Write to Delta table
songs_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.songs")
print(f"Created songs table with {songs_df.count()} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.songs").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Licenses Table
# MAGIC
# MAGIC Licenses represent the legal agreements that allow third parties (radio stations,
# MAGIC streaming services, venues) to publicly perform GMR-managed songs.

# COMMAND ----------

NUM_LICENSES = 2000

# Load songs for referencing
songs_list = [row.song_id for row in spark.table(f"{CATALOG}.{SCHEMA}.songs").collect()]

# Licensee names
LICENSEES = [
    "Spotify USA Inc.", "Apple Inc.", "Amazon.com Services LLC", "Pandora Media LLC",
    "iHeartMedia Inc.", "SiriusXM Holdings Inc.", "YouTube LLC", "TikTok Inc.",
    "Entercom Communications", "Cumulus Media Inc.", "Townsquare Media",
    "Live Nation Entertainment", "AEG Presents", "MSG Entertainment",
    "Hilton Worldwide", "Marriott International", "Starbucks Corporation",
    "Target Corporation", "Walmart Inc.", "Peloton Interactive"
]

license_data = []
for i in range(1, NUM_LICENSES + 1):
    song_id = random.choice(songs_list)
    licensee = random.choice(LICENSEES)
    license_type = random.choice(LICENSE_TYPES)
    territory = random.choice(TERRITORIES)

    # Generate license period (1-5 years)
    start_days_ago = random.randint(0, 1825)
    start_date = datetime.now() - timedelta(days=start_days_ago)
    license_years = random.randint(1, 5)
    end_date = start_date + timedelta(days=365 * license_years)

    # Fee amount varies by license type
    base_fee = {
        "streaming": 5000, "radio": 2500, "live_venue": 10000,
        "digital": 3000, "sync": 25000, "mechanical": 1500
    }
    fee = round(base_fee.get(license_type, 3000) * random.uniform(0.5, 3.0), 2)

    license_data.append({
        "license_id": f"LIC{i:07d}",
        "song_id": song_id,
        "licensee_name": licensee,
        "license_type": license_type,
        "territory": territory,
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "fee_amount": fee
    })

licenses_df = spark.createDataFrame(license_data)
licenses_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.licenses")
print(f"Created licenses table with {licenses_df.count()} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.licenses").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Performance Logs Table
# MAGIC
# MAGIC Performance logs track every time a song is played across various platforms.
# MAGIC This data is critical for calculating royalties owed to songwriters.

# COMMAND ----------

NUM_PERFORMANCE_LOGS = 50000

# Use dbldatagen for efficient large-scale generation
performance_logs_spec = (
    dg.DataGenerator(spark, name="performance_logs", rowcount=NUM_PERFORMANCE_LOGS)
    .withColumn("log_id", "string", expr="concat('LOG', lpad(cast(id as string), 8, '0'))")
    .withColumn("song_id", "string", values=songs_list, random=True)
    .withColumn("platform", "string", values=PLATFORMS, random=True)
    .withColumn(
        "play_timestamp", "timestamp",
        begin=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        random=True
    )
    .withColumn("territory", "string", values=TERRITORIES, random=True)
    .withColumn(
        "duration_played", "int",
        minValue=30, maxValue=420,  # 30 seconds to 7 minutes
        random=True
    )
    .withColumn("reported_by", "string", values=LICENSEES[:10], random=True)
)

performance_logs_df = performance_logs_spec.build()

# Write to Delta table
performance_logs_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.performance_logs")
print(f"Created performance_logs table with {performance_logs_df.count()} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.performance_logs").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Royalty Payments Table
# MAGIC
# MAGIC Royalty payments represent the financial transactions to songwriters based on
# MAGIC their song performances. Accurate payment tracking is core to GMR's mission.

# COMMAND ----------

NUM_ROYALTY_PAYMENTS = 5000

# Generate payment periods (quarterly)
payment_periods = []
for year in range(2022, 2026):
    for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        payment_periods.append(f"{year}-{quarter}")

PAYMENT_STATUSES = ["completed", "pending", "processing", "disputed", "on_hold"]

royalty_data = []
for i in range(1, NUM_ROYALTY_PAYMENTS + 1):
    song_id = random.choice(songs_list)
    songwriter_id = random.choice(songwriters_list)
    period = random.choice(payment_periods)

    # Generate realistic payment amounts
    gross_amount = round(random.uniform(50, 25000), 2)
    # Deductions typically 10-25% (admin fees, taxes, etc.)
    deduction_rate = random.uniform(0.10, 0.25)
    deductions = round(gross_amount * deduction_rate, 2)
    net_amount = round(gross_amount - deductions, 2)

    status = random.choices(
        PAYMENT_STATUSES,
        weights=[0.75, 0.10, 0.08, 0.05, 0.02]  # Most payments are completed
    )[0]

    # Payment date based on period end + processing time
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
        "payment_date": payment_date.date() if status == "completed" else None
    })

royalty_payments_df = spark.createDataFrame(royalty_data)
royalty_payments_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.royalty_payments")
print(f"Created royalty_payments table with {royalty_payments_df.count()} records")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.royalty_payments").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Summary

# COMMAND ----------

# Display summary of all created tables
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
# MAGIC 1. **02_data_ingestion.py** - Demonstrate Auto Loader patterns for incremental ingestion
# MAGIC 2. **03_feature_engineering.py** - Create Feature Store tables for ML-ready features
# MAGIC 3. **04_vector_index.py** - Build vector search index for semantic song discovery

# COMMAND ----------

print(f"""
GMR Demo Data Generation Complete!
==================================
Catalog: {CATALOG}
Schema: {SCHEMA}

Tables created:
- {CATALOG}.{SCHEMA}.songwriters ({NUM_SONGWRITERS} rows)
- {CATALOG}.{SCHEMA}.songs ({NUM_SONGS} rows)
- {CATALOG}.{SCHEMA}.licenses ({NUM_LICENSES} rows)
- {CATALOG}.{SCHEMA}.performance_logs ({NUM_PERFORMANCE_LOGS} rows)
- {CATALOG}.{SCHEMA}.royalty_payments ({NUM_ROYALTY_PAYMENTS} rows)
""")
