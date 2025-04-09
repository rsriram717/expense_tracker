"""
Centralized configuration for the financial transaction categorization system.
All constants, environment variables, and configuration parameters are defined here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
DATA_DIR = "data"
MODELS_DIR = "models"
TO_CATEGORIZE_DIR = os.path.join(DATA_DIR, "to_categorize")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MERCHANTS_DIR = os.path.join(DATA_DIR, "merchants")
MODEL_VERSION_FILE = os.path.join(DATA_DIR, "model_version.txt")

# Database configuration
DB_PATH = f'sqlite:///{os.path.join(DATA_DIR, "transactions.db")}'

# Model configuration
RF_CONFIDENCE_THRESHOLD = 0.7
MIN_MERCHANT_CONFIDENCE = 0.85

# --- Inference.net Configuration ---
INFERENCE_API_KEY_ENV_VAR = "INFERENCE_API_KEY"
INFERENCE_API_KEY = os.getenv(INFERENCE_API_KEY_ENV_VAR)
INFERENCE_BASE_URL = "https://api.inference.net/v1/"
LLAMA_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct/fp-8"

# Predefined transaction categories
PREDEFINED_CATEGORIES = [
    "Food & Drink", "Groceries", "Transportation", "Utilities", "Home",
    "Subscriptions", "Shopping", "Entertainment", "Travel-Airline", 
    "Travel-Lodging", "Medical", "Clothes", "Dante", "Misc"
]

# Merchant file path
MERCHANT_CATEGORIES_FILE = os.path.join(MERCHANTS_DIR, "merchant_categories.csv")

# Define categories with examples to help the model understand context
CATEGORY_EXAMPLES = {
    "Food & Drink": [
        "UBER EATS", "DOORDASH", "GRUBHUB", "CHIPOTLE", "STARBUCKS",
        "MCDONALD'S", "RESTAURANTS", "BARS", "COFFEE SHOPS", "FOOD DELIVERY"
    ],
    "Groceries": [
        "TRADER JOE'S", "JEWEL OSCO", "WHOLE FOODS", "MARIANO'S", "ALDI",
        "WALMART GROCERY", "TARGET GROCERIES", "KROGER"
    ],
    "Transportation": [
        "UBER", "LYFT", "METRA", "CTA", "DIVVY BIKE", "SHELL", "EXXON",
        "CHEVRON", "GAS STATIONS", "TRANSIT"
    ],
    "Utilities": [
        "COMED", "NICOR GAS", "AT&T", "VERIZON", "COMCAST", "WATER BILL",
        "PEOPLES GAS", "INTERNET", "ELECTRIC"
    ],
    "Home": [
        "HOME DEPOT", "LOWE'S", "BED BATH & BEYOND", "WAYFAIR", "IKEA",
        "FURNITURE", "RENT PAYMENT", "APARTMENT FEES", "PROPERTY MANAGEMENT"
    ],
    "Subscriptions": [
        "NETFLIX", "HULU", "SPOTIFY", "APPLE MUSIC", "DISNEY+",
        "HBO MAX", "YOUTUBE PREMIUM", "MONTHLY MEMBERSHIPS"
    ],
    "Shopping": [
        "AMAZON", "EBAY", "TARGET", "WALMART", "BEST BUY", "MACY'S",
        "NORDSTROM", "RETAIL STORES"
    ],
    "Entertainment": [
        "AMC THEATERS", "REGAL CINEMAS", "TICKETMASTER", "LIVE NATION",
        "CONCERT TICKETS", "THEME PARKS"
    ],
    "Travel-Airline": [
        "UNITED AIRLINES", "AMERICAN AIRLINES", "DELTA AIRLINES", "SOUTHWEST",
        "FRONTIER", "AIRLINE TICKETS"
    ],
    "Travel-Lodging": [
        "MARRIOTT", "HILTON", "HYATT", "AIRBNB", "HOTELS.COM",
        "EXPEDIA", "HOTEL STAYS"
    ],
    "Medical": [
        "CVS PHARMACY", "WALGREENS", "DOCTOR VISIT", "HOSPITAL PAYMENT", "DENTAL CARE",
        "INSURANCE COPAY", "MEDICAL EXPENSES"
    ],
    "Clothes": [
        "GAP", "OLD NAVY", "H&M", "ZARA", "NORDSTROM",
        "MACY'S", "CLOTHING STORES"
    ],
    "Dante": [
        "PET SUPPLIES", "PETCO", "PETSMART", "VET VISIT", "PET FOOD",
        "PET SERVICES", "PET INSURANCE", "PET GROOMING"
    ],
    "Misc": [
        "OTHER TRANSACTIONS", "PAYMENT METHOD AUTHORIZATION", "UNRECOGNIZED MERCHANT",
        "PAYPAL PAYMENT", "BANK FEE", "VENMO", "CASH APP"
    ]
} 